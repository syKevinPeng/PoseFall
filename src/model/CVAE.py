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


class CAVE(nn.Module):
    def __init__(self, phase_names, num_classes_dict, device = "cuda") -> None:
        super().__init__()
        self.phase_names = phase_names
        self.num_classes_dict = num_classes_dict
        self.device = device
        # initialize encoder and decoder for each phase
        for phase in self.phase_names:
            setattr(
                self,
                f"{phase}_encoder",
                Encoder(num_classes=num_classes_dict[phase], phase_names=phase),
            )
            setattr(
                self,
                f"{phase}_decoder",
                Decoder(num_classes=num_classes_dict[phase], phase_names=phase),
            )

        print(f"CAVE model initialized with phases: {self.phase_names}")

    def reparameterize(self, batch, phase_name, seed=None):
        """
        reparameterize the latent variable
        """
        mu, sigma = batch[f"{phase_name}_mu"], batch[f"{phase_name}_sigma"]
        std = torch.exp(sigma / 2)
        if seed:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)
        else:
            eps = std.data.new(std.size()).normal_()
        eps = torch.randn_like(sigma)
        return mu + eps * sigma  # eps.mul(std).add_(mu)

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

        # fuse the encoder output by concate output layers and project to 3 mus and 3 sigmas
        concated_output = torch.cat(encoder_output_list, dim=1).to(self.device)
        fuse_layer = nn.Linear(concated_output.size(1), concated_output.size(1)).to(self.device)
        fused_output = fuse_layer(concated_output).reshape(batch_size, 6, -1)
        # assign mu and sigma back to the batch
        for i, phase in enumerate(self.phase_names):
            batch[f"{phase}_mu"] = fused_output[:, 2*i, :]
            batch[f"{phase}_sigma"] = fused_output[:, 2*i+1, :]
            
        for phase in self.phase_names:
            # reparameterize
            batch[f"{phase}_z"] = self.reparameterize(batch, phase_name=phase)
        for phase in self.phase_names:
            # Decoder
            batch.update(getattr(self, f"{phase}_decoder")(batch))
        return batch

    # def return_latent(self, batch):
    #     # encode
    #     batch.update(self.encoder(batch))
    #     # reparameterize
    #     batch[f"z"] = self.reparameterize(batch)
    #     return batch["z"]

    def compute_loss(self, batch, phase_name):
        pred_batch = torch.clone(batch[f"{phase_name}_output"])
        input_batch = torch.clone(batch[f"{phase_name}_combined_poses"])
        mask_batch = batch[f"{phase_name}_src_key_padding_mask"]

        padding = mask_batch.bool().unsqueeze(-1).expand(-1, -1, pred_batch.size(-1))
        input_batch[padding] = 0
        pred_batch[padding] = 0

        # human model param l2 loss
        human_model_loss = human_param_loss(pred_batch, input_batch)

        # KL divergence loss
        mu, logvar = batch[f"{phase_name}_mu"], batch[f"{phase_name}_sigma"]
        kl_loss = kl_divergence(mu, logvar)

        # vertex loss
        vertex_locs_loss = vertex_loss(pred_batch, input_batch)
        # loss weight
        loss_weight = {
            "human_model_loss": 1,
            "kl_loss": 1e-5,
            "vertex_loss": 1,
        }
        # compute loss
        total_phase_loss = (
            loss_weight["human_model_loss"] * human_model_loss
            + loss_weight["kl_loss"] * kl_loss
            + loss_weight["vertex_loss"] * vertex_locs_loss
        )
        # log corresponding loss
        # print(
        #     f"{phase_name} human model loss: {human_model_loss.item()}, kl loss: {kl_loss.item()}, vertex loss: {vertex_locs_loss.item()}"
        # )

        return total_phase_loss
    
    def compute_all_phase_loss(self, batch):
        total_loss = 0
        for phase in self.phase_names:
            total_loss += self.compute_loss(batch, phase)
        return total_loss
