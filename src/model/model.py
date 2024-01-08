"""
This file contains the encoder and decoder for the transformer model. The actual model is in CVAE.py 
"""
from cmath import phase
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from icecream import ic
torch.manual_seed(0)

# standard transformer Positional encoding from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        x = self.dropout(x)
        return x


class  Encoder(nn.Module):
    """
    Encoder for the transformer model
    modified from https://github.com/Mathux/ACTOR/blob/master/src/models/architectures/transformer.py
    """

    def __init__(
        self,
        num_classes,
        phase_names,
        input_feature_dim=150,
        latent_dim=256,
        num_att_layers=8,
        num_heads=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="gelu",
    ):
        super(Encoder, self).__init__()
        self.phase_names = phase_names
        self.latent_dim = latent_dim  # latent dimension
        self.num_classes = num_classes
        # transformer parameters
        self.num_heads = num_heads  # attention heads
        self.dim_feedforward = (
            dim_feedforward  # dimension of the feedforward network (fc layers)
        )
        self.dropout = dropout  # dropout rate
        self.activation = activation  # activation function
        self.num_att_layers = num_att_layers
        self.input_feature_dim = input_feature_dim

        # motion distribution parameter tokens: used for pooling the temporal dimension
        self.muQuery = nn.Parameter(
            torch.randn(self.num_classes, self.latent_dim, dtype=torch.float32)
        )

        self.sigmaQuery = nn.Parameter(
            torch.randn(self.num_classes, self.latent_dim, dtype=torch.float32)
        )

        # standard transformer encoder
        # define one layer
        trans_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            # batch_first=True,
        )
        # define the entire encoder
        self.seqTransEncoder = nn.TransformerEncoder(
            trans_layer, num_layers=self.num_att_layers
        )

        self.sequence_pos_encoder  = PositionalEncoding(latent_dim)

        self.skelEmbedding = nn.Linear(self.input_feature_dim, self.latent_dim)

    def forward(self, batch):
        """
        Arguments:
            data: Tensor, shape ``[batch_size, seq_len, feature_dim]``
            label: Tensor, shape ``[batch_size, num_classes]``
            mask: Tensor, shape ``[batch_size, seq_len, feature_dim]``
        """
        data, label, mask = (
            batch[f"{self.phase_names}_combined_poses"].permute(1,0,2), # permute to frame_num, batch_size, joint_num*feat_dim
            batch[f"{self.phase_names}_label"],
            batch[f"{self.phase_names }_src_key_padding_mask"].bool(), # permute to frame_num, batch_size
        )
        batch_size = data.size(1)
        # human poses embedding
        x:Tensor = self.skelEmbedding(data) 
        # add mu and sigma queries
        # select where the label is 1
        muQuery = torch.matmul(label, self.muQuery).unsqueeze(1).permute(1, 0, 2)
        sigmaQuery = torch.matmul(label, self.sigmaQuery).unsqueeze(1).permute(1, 0, 2)
        # add mu and sigma queries to the input
        xseq = torch.cat((muQuery, sigmaQuery, x), dim=0) # shape: time+2, bs, latent_dim
        # add positional encoding
        encoded_xseq = self.sequence_pos_encoder(xseq)
        # create a bigger mask to attend to mu and sigma
        extra_mask = torch.zeros((batch_size, 2), dtype=bool).to(x.device)
        mask_seq = torch.cat((extra_mask, mask), axis=1).bool()
        encoder_output = self.seqTransEncoder(
            encoded_xseq, src_key_padding_mask=mask_seq
        )
        # get the first two output
        mu = encoder_output[0]
        sigma = encoder_output[1]
        return {f"{self.phase_names}_mu": mu, f"{self.phase_names}_sigma": sigma}
        
        # instead of returning mu and sigma, return the entire encoder output
        # ic(encoder_output.size())
        # return {f"{self.phase_names}_encoder_output": encoder_output}

class Decoder(nn.Module):
    """
    Decoder for the transformer model
    modified from
    """

    def __init__(
        self,
        num_classes,
        phase_names,
        input_feature_dim=150,
        latent_dim=256,
        num_heads=4,
        dim_feedforward=1024,
        num_layers=8,
        dropout=0.1,
        activation="gelu",
    ) -> None:
        super().__init__()
        self.phase_names = phase_names
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(latent_dim)
        self.input_feats = input_feature_dim

        self.action_biases = nn.Parameter(
            torch.randn(self.num_classes, self.latent_dim)
        )

        self.seqTransDecoderLayer  = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            self.seqTransDecoderLayer , num_layers=self.num_layers
        )
        self.finallayer = nn.Linear(in_features=self.latent_dim, out_features=self.input_feats)
        if self.phase_names == "combined":
            self.combined = True
        else:
            self.combined = False

    def forward(self, batch):
        if self.combined:
            z = batch["combined_z"]
            y = batch["combined_label"]
            mask = batch["combined_src_key_padding_mask"]
        z = batch[f"{self.phase_names}_z"]
        y = batch[f"{self.phase_names}_label"]
        mask = batch[f"{self.phase_names}_src_key_padding_mask"]
        latent_dim = z.size(1)
        batch_size, num_frames = mask.shape
        # shift the latent noise vector to be the action noise
        shifted_z = z + y @ self.action_biases
        # z is sequence of size 1
        shifted_z = shifted_z.unsqueeze(0)
        # time queries
        timequeries = torch.zeros(
            num_frames, batch_size, latent_dim, device=shifted_z.device
        )
        # sequence encoding
        timequeries = self.sequence_pos_encoder(timequeries)
        # decode
        decoder_output = self.seqTransDecoder(
            tgt=timequeries, memory=shifted_z, tgt_key_padding_mask=mask.bool()
        )
        # get output sequences
        output = self.finallayer(decoder_output).reshape(batch_size, num_frames, -1)

        # setting zero for padded area
        # expand the mask to the output size
        # padding = mask.bool().unsqueeze(-1).expand(-1, -1, output.size(-1))
        # output[padding] = 0
        batch[f"{self.phase_names}_output"] = output
        if self.combined:
            batch["output"] = output
        return batch
