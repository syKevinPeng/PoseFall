from ast import parse
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import *
from icecream import ic




class CAVE(nn.Module):
    def __init__(self,encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, batch, seed = None):
        """
        reparameterize the latent variable
        """
        mu, sigma = batch['mu'], batch['sigma']
        std = torch.exp(sigma / 2)
        if seed:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)
        else:
            eps = std.data.new(std.size()).normal_()
        eps = torch.randn_like(sigma)
        return mu + eps * sigma #eps.mul(std).add_(mu)
    
    def forward(self, batch):
        # encode
        batch.update(self.encoder(batch))
        # reparameterize
        batch["z"] = self.reparameterize(batch)
        # decode
        batch.update(self.decoder(batch))

        return batch
    
    def return_latent(self, batch):
        # encode
        batch.update(self.encoder(batch))
        # reparameterize
        batch["z"] = self.reparameterize(batch)
        return batch["z"]
    
    def compute_loss(self, batch):
        pred_batch = batch["output"]
        input_batch = torch.clone(batch["data"])
        mask_batch = batch["mask"]

        padding = mask_batch.bool().unsqueeze(-1).expand(-1, -1, pred_batch.size(-1))
        input_batch[padding] = 0
        pred_batch[padding] = 0

        ic(input_batch)
        ic(mask_batch)

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
        total_loss = \
            loss_weight["human_model_loss"] * human_model_loss +\
            loss_weight["kl_loss"] * kl_loss +\
            loss_weight["vertex_loss"] * vertex_locs_loss
        
        return total_loss
