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

class CVAE3E3D(nn.Module):
    """
    CVAE model with three decoder and three encoders.
    """
    def __init__(self, data_config_dict:dict, config, latent_dim = 256, device = "cuda") -> None:
        super().__init__()
        self.phase_names = config['constant']['PHASES']
        self.data_config_dict = data_config_dict
        self.device = device
        self.latent_dim =latent_dim
        self.num_joints = data_config_dict["num_joints"]
        self.feat_dim = data_config_dict["feat_dim"]
        # initialize encoder and decoder for each phase
        for phase in self.phase_names:
            setattr(
                self,
                f"{phase}_encoder",
                Encoder(num_classes=data_config_dict[phase]["label_size"], 
                        phase_names=phase, 
                        latent_dim=self.latent_dim, 
                        njoints=data_config_dict["num_joints"], 
                        nfeats=data_config_dict["feat_dim"],
                        input_feature_dim=self.num_joints*self.feat_dim)
            )
            setattr(
                self,
                f"{phase}_decoder",
                Decoder(num_classes=data_config_dict[phase]["label_size"], 
                        phase_names=phase, 
                        latent_dim=self.latent_dim, 
                        njoints=data_config_dict["num_joints"], 
                        nfeats=data_config_dict["feat_dim"],
                        input_feature_dim=self.num_joints*self.feat_dim)
            )
        self.config = config
        self.fusion_layer = nn.Linear(in_features=self.latent_dim*len(self.phase_names), out_features=self.latent_dim*len(self.phase_names))
        print(f"CAVE model initialized with phases: {self.phase_names}")

    # def reparameterize(self, batch, phase_name, seed=0):
    #     """
    #     reparameterize the latent variable
    #     """
    #     combined_mu = torch.cat([batch[f"{phase}_mu"] for phase in self.phase_names], dim=1)
    #     combined_sigma = torch.cat([batch[f"{phase}_sigma"] for phase in self.phase_names], dim=1)
    #     # remove phase_mu and phase_sigma from batch
    #     for phase in self.phase_names:
    #         batch.pop(f"{phase}_mu")
    #         batch.pop(f"{phase}_sigma")
    #     batch.update({"combined_mu": combined_mu, "combined_sigma": combined_sigma})
    #     std = torch.exp(combined_sigma / 2)

    #     if seed is None:
    #         eps = std.data.new(std.size()).normal_()
    #     else:
    #         generator = torch.Generator(device=self.device)
    #         generator.manual_seed(seed)
    #         eps = std.data.new(std.size()).normal_(generator=generator)

    #     z = eps.mul(std).add_(combined_mu)
    #     return z
    def reparameterize(self, batch, phase_name, seed=0):
        """
        reparameterize the latent variable
        """
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

    def prepare_batch(self, batch):
        for phase in self.phase_names:
            batch[f"{phase}_combined_poses"] = batch[f"{phase}_combined_poses"].to(self.device)
            batch[f"{phase}_label"] = batch[f"{phase}_label"].to(self.device)
            batch[f"{phase}_src_key_padding_mask"] = batch[f"{phase}_src_key_padding_mask"].to(self.device)
            # batch[f"{phase}_lengths"] = batch[f"{phase}_lengths"].to(self.device)
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

        # # fuse the encoder output by concate output layers and project to 3 mus and 3 sigmas
        # concated_output = torch.cat(encoder_output_list, dim=1).to(self.device)
        # fuse_layer = nn.Linear(concated_output.size(1), concated_output.size(1)).to(self.device)
        # fused_output = fuse_layer(concated_output).reshape(batch_size, 6, -1)
        # # assign mu and sigma back to the batch
        # for i, phase in enumerate(self.phase_names):
        #     batch[f"{phase}_mu"] = fused_output[:, 2*i, :]
        #     batch[f"{phase}_sigma"] = fused_output[:, 2*i+1, :]

        for phase in self.phase_names:
            # reparameterize
            batch[f"{phase}_z"] = self.reparameterize(batch, phase_name=phase) # shape(batch_size, latent_dim)
        # Fuse the latent variable
        combined_z = torch.cat([batch[f"{phase}_z"] for phase in self.phase_names], dim=1)
        fused_z = self.fusion_layer(combined_z)
        # split the fused_z into three parts
        for i, phase in enumerate(self.phase_names):
            batch[f"{phase}_z"] = fused_z[:, self.latent_dim*i:self.latent_dim*(i+1)]
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
    

    def compute_loss(self, batch):
        total_loss = 0
        all_loss_dict = {}
        for phase in self.phase_names:   
            loss, in_phase_loss_dict = compute_in_phase_loss(batch, phase_name = phase, weight_dict=self.config['loss_config'])
            total_loss += loss
            all_loss_dict.update(in_phase_loss_dict)
        interphase_loss, inter_loss_dict = compute_inter_phase_loss(self.phase_names, batch, loss_weights_dict=self.config['loss_config'])
        total_loss += interphase_loss
        all_loss_dict.update(inter_loss_dict)
        return total_loss, all_loss_dict
    
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

        # remove the padding based on the mask
        for phase in self.phase_names:
            model_input_batch[f"{phase}_output"] = model_input_batch[f"{phase}_output"]
            # remove the padding
            model_output = model_input_batch[f"{phase}_output"]
            # filter out the positions where the mask is 1
            mask = model_input_batch[f"{phase}_src_key_padding_mask"] # mask: 0 means valid, 1 means invalid
            # inverse the mask
            mask = 1-mask
            mask = mask.unsqueeze(-1).expand(-1, -1, model_output.size(-1))
            # remove the padding
            model_output = model_output * mask
            # remove the rows where all the elements are 0. 
            model_output = model_output[model_output.sum(dim=2) != 0]
            # check the dimension
            if len(model_output.size()) == 2: # If the batch size is 1, add a dimension
                model_output = model_output.unsqueeze(0)
            model_input_batch[f"{phase}_output"] = model_output
        return model_input_batch
