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
import wandb

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
        pred_batch = batch[f"{phase_name}_output"]
        input_batch = batch[f"{phase_name}_combined_poses"]
        mask_batch = batch[f"{phase_name}_src_key_padding_mask"]

        padding = ~(mask_batch.bool().unsqueeze(-1).expand(-1, -1, pred_batch.size(-1)))
        pred_batch = pred_batch * padding
        input_batch = input_batch * padding

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
        return total_phase_loss
    
    def compute_inter_phase_loss(self, batch):
        """
        take 10% of the end of the first phase and 10% of the beginning of the second phase
        compute the first derivative of the joint location        
        """
        inter_phase_loss = 0
        for i in range(len(self.phase_names)-1):
            first_phase = self.phase_names[i]
            second_phase = self.phase_names[i+1]
            pred_first_phase = batch[f"{first_phase}_output"]
            pred_second_phase = batch[f"{second_phase}_output"]
            # take the last 10% of the first phase
            first_phase_last_10 = int(pred_first_phase.size(1) * 0.1)
            # take the first 10% of the second phase
            second_phase_first_10 = int(pred_second_phase.size(1) * 0.1)
            # concatenate the two phase
            combined_pred = torch.cat((pred_first_phase[:, -first_phase_last_10:, :], pred_second_phase[:, :second_phase_first_10, :]), dim=1)
            # get the joint locations using SMPL model
            smpl_model = SMPLModel().eval().to(DEVICE)
            # get the vertex locations
            _, joint_locs = smpl_model(combined_pred)
            # take the first 24 joints (there are only 24 joints for SMPL model but they author make it more to fit for other models)
            joint_locs = joint_locs[:, :, :24, :]
            # calculate the first derivative of the joint locations
            joint_locs_diff = joint_locs[:, 1:, :, :] - joint_locs[:, :-1, :, :]
            # calculate the variance across the time
            joint_locs_diff_var = torch.var(joint_locs_diff, dim=1)
            # calculate the mean of the variance
            inter_phase_loss += torch.mean(joint_locs_diff_var)
        return inter_phase_loss


    def compute_all_phase_loss(self, batch):
        total_loss = 0
        for phase in self.phase_names:
            total_loss += self.compute_loss(batch, phase)
        interphase_loss = self.compute_inter_phase_loss(batch)
        total_loss += interphase_loss
        return total_loss
