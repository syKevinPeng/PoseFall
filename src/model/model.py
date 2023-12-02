import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np


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

        # motion distribution parameter tokens: used for pooling the temporal dimension
        self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # standard transformer encoder
        # define one layer
        trans_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
        # define the entire encoder
        trans_encoder = nn.TransformerEncoder(
            trans_layer, num_layers=self.num_att_layers
        )

        self.pos_encoder = PositionalEncoding(latent_dim)

    def forward(self, batch):
        data, label, mask = batch["data"], batch["label"], batch["mask"]
        batch_size, num_joints, num_features, num_frames = data.shape
