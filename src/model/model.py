import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from icecream import ic


# standard transformer Positional encoding from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Encoder(nn.Module):
    """
    Encoder for the transformer model
    modified from https://github.com/Mathux/ACTOR/blob/master/src/models/architectures/transformer.py
    """

    def __init__(
        self,
        num_classes,
        input_feature_dim=153,
        latent_dim=256,
        num_att_layers=8,
        num_heads=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="gelu",
    ): 
        super(Encoder, self).__init__()
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
        self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim, dtype=torch.float32))

        self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim, dtype=torch.float32))

        # standard transformer encoder
        # define one layer
        trans_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        # define the entire encoder
        self.trans_encoder = nn.TransformerEncoder(
            trans_layer, num_layers=self.num_att_layers
        )

        self.pos_encoder = PositionalEncoding(latent_dim)

        self.skelEmbedding = nn.Linear(self.input_feature_dim, self.latent_dim)

    def forward(self, batch):
        """
        Arguments:
            data: Tensor, shape ``[batch_size, seq_len, feature_dim]``
            label: Tensor, shape ``[batch_size, num_classes]``
            mask: Tensor, shape ``[batch_size, seq_len, feature_dim]``
        """
        data, label, mask = batch['data'], batch['label'], batch['mask']
        batch_size = data.size(0)
        # human poses embedding
        x = self.skelEmbedding(data)
        
        # add mu and sigma queries
        # select where the label is 1
        muQuery = torch.matmul(label, self.muQuery).unsqueeze(1)
        sigmaQuery = torch.matmul(label, self.sigmaQuery).unsqueeze(1)
        # add mu and sigma queries to the input
        xseq = torch.cat((muQuery, sigmaQuery, x), dim=1)
        # add positional encoding
        encoded_xseq = self.pos_encoder(xseq) 
        
        # create a bigger mask to attend to mu and sigma
        extra_mask = torch.zeros((batch_size, 2)).to(mask.device)
        mask_seq = torch.cat((extra_mask, mask), dim=1)
        
        encoder_output = self.trans_encoder(encoded_xseq, src_key_padding_mask=mask_seq.bool())
        # get the first two output
        mu = encoder_output[:, 0, :]
        sigma = encoder_output[:, 1, :]
        return {'mu': mu, 'sigma': sigma}
    
class Decoder(nn.Module):
    """
    Decoder for the transformer model
    modified from
    """
    def __init__(self,num_classes, input_feature_dim=153, latent_dim = 256, num_heads = 4, dim_feedforward = 1024, num_layers = 4, dropout=0.1, activation="gelu") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim  
        self.num_heads = num_heads
        self.dim_feedforward=dim_feedforward
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.pos_encoder = PositionalEncoding(latent_dim)
        self.input_feats = input_feature_dim

        self.action_biases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.trans_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )

        self.trans_decoder = nn.TransformerDecoder(self.trans_layer, 
                                                   num_layers=self.num_layers)
        self.final_layer = nn.Linear(self.latent_dim, self.input_feats)

    
    def forward(self, batch):
        z, y, mask, lengths = batch['z'], batch['label'], batch['mask'], batch['lengths']
        latent_dim = z.size(1)
        batch_size, num_frames = mask.shape
        # shift the latent noise vector to be the action noise
        shifted_z = z + y @ self.action_biases
        # z is sequence of size 1
        shifted_z = shifted_z.unsqueeze(1)
        # time queries
        timequeries = torch.zeros(batch_size, num_frames, latent_dim, device=shifted_z.device)
        # sequence encoding
        timequeries = self.pos_encoder(timequeries)

        # decode
        decoder_output = self.trans_decoder(tgt = timequeries, memory=shifted_z, tgt_key_padding_mask=mask.bool())
        # get output sequences
        output = self.final_layer(decoder_output).reshape(batch_size, num_frames, -1)

        # setting zero for padded area
        # expand the mask to the output size
        # padding = mask.bool().unsqueeze(-1).expand(-1, -1, output.size(-1))
        # output[padding] = 0
        batch["output"] = output
        return batch