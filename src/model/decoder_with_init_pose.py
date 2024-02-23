from .model import PositionalEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodeWithInitPose(nn.Module):
    """
    Decoder for the transformer model
    modified from https://github.com/Mathux/ACTOR/blob/d3b0afe674e01fa2b65c89784816c3435df0a9a5/src/models/architectures/transformer.py#L131
    Added the initial pose as input to the decoder
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

    def forward(self, batch):
        z = batch[f"{self.phase_names}_z"]
        y = batch[f"{self.phase_names}_label"]
        init_pose = batch[f"{self.phase_names}_init_pose"]
        mask = batch[f"{self.phase_names}_src_key_padding_mask"]
        mask = mask.bool()
        latent_dim = z.size(1)
        bs, nframes = mask.shape

        # shift the latent noise vector to be the action noise
        shifted_z = z + y @ self.actionBiases


        print(f'init pose size: {init_pose.size()}')
        print(f'z size: {z.size()}')
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
