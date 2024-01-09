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
    def _generate(self, batch, nspa=1,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1):
        if nspa is None:
            nspa = 1
        batch_size = batch["combined_label"].size(0)
        if batch_size != 1:
            raise Exception("batch size must be 1")
        classes = batch["combined_label"]
        mask = batch["combined_mask"]
        mask = 1-mask
        durations = mask.sum(dim=1).int()
        # cast to int
        nats = len(classes)
        # y = classes.to(self.device).repeat(nspa)  # (view(nspa, nats))
        y = classes.to(self.device) 

        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(y.shape)
        
        mask = self.lengths_to_mask(lengths)
        if noise_same_action == "random":
            if noise_diff_action == "random":
                # here
                z = torch.randn(nspa*nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_same_action = torch.randn(nspa, self.latent_dim, device=self.device)
                z = z_same_action.repeat_interleave(nats, axis=0)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
        elif noise_same_action == "interpolate":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            interpolation_factors = torch.linspace(-1, 1, nspa, device=self.device)
            z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa*nats, -1)
        elif noise_same_action == "same":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            z = z_diff_action.repeat((nspa, 1))
        else:
            raise NotImplementedError("Noise same action must be random, same or interpolate.")
        input_batch = {"combined_z": fact*z, "combined_label": y, "combined_src_key_padding_mask": mask, "lengths": lengths}
        output = self.decoder(input_batch)
        batch.update(output)
        
        # if self.outputxyz:
        #     batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        # elif self.pose_rep == "xyz":
        #     batch["output_xyz"] = batch["output"]
        
        return batch

    def lengths_to_mask(self, lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return ~mask
