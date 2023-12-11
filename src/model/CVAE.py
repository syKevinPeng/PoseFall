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
        prepared_batch = {}
        for phase in self.phase_names:
            prepared_batch[phase] = {
                f"{phase}_data": batch[f"{phase}_combined_poses"].to(self.device),
                f"{phase}_label": batch[f"{phase}_label"].to(self.device),
                f"{phase}_mask": batch[f"{phase}_src_key_padding_mask"].to(self.device),
                f"{phase}_length": batch[f"{phase}_lengths"].to(self.device),
            }
        return prepared_batch

    def forward(self, batch):
        prepared_batch = self.prepare_batch(batch)
        for phase in self.phase_names:
            # Encoder
            batch.update(getattr(self, f"{phase}_encoder")(prepared_batch[phase]))
            # reparameterize
            batch[f"f{phase}_z"] = self.reparameterize(prepared_batch[phase], phase_name = phase)
            # Decoder
            batch.update(getattr(self, f"{phase}_decoder")(prepared_batch[phase]))
        return batch

    # def return_latent(self, batch):
    #     # encode
    #     batch.update(self.encoder(batch))
    #     # reparameterize
    #     batch[f"z"] = self.reparameterize(batch)
    #     return batch["z"]

    def compute_loss(self, batch):
        pred_batch = batch["output"]
        input_batch = torch.clone(batch["data"])
        mask_batch = batch["mask"]

        padding = mask_batch.bool().unsqueeze(-1).expand(-1, -1, pred_batch.size(-1))
        input_batch[padding] = 0
        pred_batch[padding] = 0

        # human model param l2 loss
        human_model_loss = human_param_loss(pred_batch, input_batch)

        # KL divergence loss
        mu, logvar = batch["mu"], batch["sigma"]
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
        total_loss = (
            loss_weight["human_model_loss"] * human_model_loss
            + loss_weight["kl_loss"] * kl_loss
            + loss_weight["vertex_loss"] * vertex_locs_loss
        )

        return total_loss
