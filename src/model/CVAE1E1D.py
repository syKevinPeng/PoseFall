"""
This file contains CVAE model. Refer to model.py for the encoder and decoder
"""
from ast import parse
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import *
from icecream import ic
from .model import Encoder, Decoder
import wandb, scipy

class CVAE1E1D(nn.Module):
    def __init__(self, num_classes, config, latent_dim = 256, phase_names = "combined", device = "cuda") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.latent_dim =latent_dim
        self.phase_names = phase_names
        self.config = config
        self.num_frames = num_classes["num_frames"]
        self.num_joints = num_classes["num_joints"]
        self.feat_dim = num_classes["feat_dim"]
        self.encoder = Encoder(num_classes=num_classes[phase_names],phase_names="combined", latent_dim=self.latent_dim, input_feature_dim=self.num_joints*self.feat_dim, njoints=self.num_joints, nfeats=self.feat_dim)
        self.decoder = Decoder(num_classes=num_classes[phase_names],phase_names="combined", latent_dim=self.latent_dim, input_feature_dim=self.num_joints*self.feat_dim, njoints=self.num_joints, nfeats=self.feat_dim )

    def forward(self, batch):
        # input size:
        self.batch_size = batch["combined_combined_poses"].size(0)
        self.prepare_batch(batch)
        # Encoder:
        batch.update(self.encoder(batch))
        # reparameterize
        batch["combined_z"] = self.reparameterize(batch, "combined")
        # Decoder:
        batch.update(self.decoder(batch))
        return batch
    
    def prepare_batch(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
    
    def reparameterize(self, batch, phase_name, seed=0):
        """
        reparameterize the latent variable
        """
        # mu, sigma = batch[f"{phase_name}_mu"], batch[f"{phase_name}_sigma"]
        # std = torch.exp(sigma / 2)
        # if seed:
        #     generator = torch.Generator(device=self.device)
        #     generator.manual_seed(seed)
        #     eps = std.data.new(std.size()).normal_(generator=generator)
        # else:
        #     eps = std.data.new(std.size()).normal_()
        # eps = torch.randn_like(sigma)
        mu, logvar = batch[f"{phase_name}_mu"], batch[f"{phase_name}_sigma"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z
    
    def compute_loss(self,batch):
        return compute_in_phase_loss(batch, phase_name = self.phase_names, weight_dict=self.config['loss_config'])
    
    def generate(self, batch, seed=None):
        """
        generate the output from the latent variable
        """
        batch_size = batch["combined_label"].size(0)
        input_batch = {
            "combined_label": batch["combined_label"],
            "combined_src_key_padding_mask": batch["combined_mask"],
                       }
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        input_batch["combined_z"] = z
        batch.update(self.decoder(input_batch))
        return batch