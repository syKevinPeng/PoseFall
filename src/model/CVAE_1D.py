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
from model.model import Encoder, Decoder
import wandb, scipy

class CVAE1D(nn.Module):
    """
    CVAE model with three encoder and one decoders. 
    """
    def __init__(self, phase_names, num_classes_dict:dict, latent_dim = 256, device = "cuda") -> None:
        super().__init__()
        self.phase_names = phase_names
        self.num_classes_dict = num_classes_dict
        self.device = device
        self.latent_dim =latent_dim
        # initialize encoder and decoder for each phase
        for phase in self.phase_names:
            setattr(
                self,
                f"{phase}_encoder",
                Encoder(num_classes=num_classes_dict[phase], phase_names=phase, latent_dim=self.latent_dim),
            )
        self.decoder = Decoder(sum(num_classes_dict.values()), phase_names, latent_dim=self.latent_dim*3)

        print(f"CAVE model initialized with phases: {self.phase_names}")

    def reparameterize(self, batch, seed=None): 
        """
        reparameterize the latent variable: Multi variate Gaussian
        """
        reparameterized = []
        for phase in self.phase_names:
            mu, sigma = batch[f"{phase}_mu"], batch[f"{phase}_sigma"]
            std = torch.exp(sigma / 2)
            if seed:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                eps = std.data.new(std.size()).normal_(generator=generator)
            else:
                eps = std.data.new(std.size()).normal_()
            eps = torch.randn_like(sigma)
            curr_repara = mu + eps * sigma  # eps.mul(std).add_(mu)
            reparameterized.append(curr_repara)
        return torch.concat(reparameterized, dim=1)

    def prepare_batch(self, batch):
        for phase in self.phase_names:
            batch[f"{phase}_combined_poses"] = batch[f"{phase}_combined_poses"].to(self.device)
            batch[f"{phase}_label"] = batch[f"{phase}_label"].to(self.device)
            batch[f"{phase}_src_key_padding_mask"] = batch[f"{phase}_src_key_padding_mask"].to(self.device)
            batch[f"{phase}_lengths"] = batch[f"{phase}_lengths"].to(self.device)
        return batch

    def forward(self, batch):
        batch_size = batch[f"{self.phase_names[0]}_combined_poses"].size(0)
        batch = self.prepare_batch(batch)
        encoder_output_list = []
        for phase in self.phase_names:
            # Encoder
            encoder_output = getattr(self, f"{phase}_encoder")(batch)
            encoder_output_list.extend([encoder_output[f"{phase}_mu"], encoder_output[f"{phase}_sigma"]])
            batch.update(encoder_output)
        # reparameterize
        batch[f"z"] = self.reparameterize(batch) # shape(batch_size, latent_dim)
        # decoder
        batch.update(self.decoder(batch))
        return batch

    # def return_latent(self, batch):
    #     # encode
    #     batch.update(self.encoder(batch))
    #     # reparameterize
    #     batch[f"z"] = self.reparameterize(batch)
    #     return batch["z"]
    

    def compute_all_phase_loss(self, batch):
        total_loss = 0 
        total_loss += compute_in_phase_loss(batch, phase_name=None)
        return total_loss
    
    def generate(self, input_batch):
        """
        generate a sequence of poses given the input
        @param input_batch: a dictionary of input batch, it should contains the following keys: impa_label, glit_label, fall_label, impa_length, glit_length, fall_length
        """
        impa_label = input_batch["impa_label"]
        glit_label = input_batch["glit_label"]
        fall_label = input_batch["fall_label"]
        impa_mask  = input_batch["impa_mask"]
        glit_mask  = input_batch["glit_mask"]
        fall_mask  = input_batch["fall_mask"]

        batch_size = impa_label.size(0)

        model_input_batch = {}
        for phase in self.phase_names:
            # sample noise
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            model_input_batch[f"{phase}_z"] = z
            model_input_batch[f"{phase}_label"] = input_batch[f"{phase}_label"]
            model_input_batch[f"{phase}_src_key_padding_mask"] = input_batch[f"{phase}_mask"]
        
        for phase in self.phase_names: 
            # Decoder
            model_input_batch.update(getattr(self, f"{phase}_decoder")(model_input_batch))
        return model_input_batch
