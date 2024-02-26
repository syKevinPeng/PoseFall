"""
This file contains the encoder and decoder for the transformer model. The actual model is in CVAE.py 
"""
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
        njoints,
        nfeats,
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
        self.num_joints = njoints
        self.feat_dim = nfeats
        # transformer parameters
        self.num_heads = num_heads  # attention heads
        self.dim_feedforward = (
            dim_feedforward  # dimension of the feedforward network (fc layers)
        )
        self.dropout = dropout  # dropout rate
        self.activation = activation  # activation function
        self.num_att_layers = num_att_layers
        self.input_feats = input_feature_dim

        self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.dim_feedforward,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_att_layers)

    def forward(self, batch):
        """
        Arguments:
            data: Tensor, shape ``[batch_size, seq_len, feature_dim]``
            label: Tensor, shape ``[batch_size, num_classes]``
            mask: Tensor, shape ``[batch_size, seq_len, feature_dim]``
        """
        x, y, mask = (
            batch[f"{self.phase_names}_combined_poses"],
            batch[f"{self.phase_names}_label"],
            batch[f"{self.phase_names }_src_key_padding_mask"].bool(), # permute to frame_num, batch_size
        )
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_joints*self.feat_dim).permute(1, 0, 2)# permute to frame_num, batch_size, joint_num*feat_dim
        # # human poses embedding
        x :torch.Tensor= self.skelEmbedding(x)
        muQuery = torch.matmul(y, self.muQuery).unsqueeze(0)
        sigmaQuery = torch.matmul(y, self.sigmaQuery).unsqueeze(0)
        # add mu and sigma queries to the input
        xseq = torch.cat((muQuery, sigmaQuery, x), dim=0)
        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)
        # create a bigger mask, to allow attend to mu and sigma
        # muandsigmaMask shape: 20, 2; mask shape: 20, 60
        muandsigmaMask = torch.zeros((batch_size, 2), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=maskseq)
        mu = final[0]
        logvar = final[1]
        return {f"{self.phase_names}_mu": mu, f"{self.phase_names}_sigma": logvar}

class Decoder(nn.Module):
    """
    Decoder for the transformer model
    modified from
    """

    def __init__(
        self,
        num_classes,
        phase_names,
        njoints,
        nfeats,
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
        self.njoints = njoints
        self.nfeats = nfeats
        self.actionBiases = nn.Parameter(
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
        mask = mask.bool()
        latent_dim = z.size(1)
        bs, nframes = mask.shape

        # shift the latent noise vector to be the action noise
        shifted_z = z + y @ self.actionBiases

        z = shifted_z[None]  # sequence of size 1
            
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        # only for ablation / not used in the final model
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=mask)
        
        output = self.finallayer(output).reshape(nframes, bs, self.njoints, self.nfeats)
        
        # zero for padded area
        output[mask.T] = 0
        output = output.permute(1, 0, 2, 3)

        # if self.combined:
        #     batch["output"] = output
        # else:
        batch[f"{self.phase_names}_output"] = output
        return batch
