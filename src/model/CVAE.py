from ast import parse
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .smpl_model import SMPLModel
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
        input_batch = batch["data"]
        mask_batch = batch["mask"]

        ic(pred_batch.shape)
        ic(input_batch.shape)
        ic(mask_batch.shape)

        padding = mask_batch.bool().unsqueeze(-1).expand(-1, -1, pred_batch.size(-1))
        input_batch[padding] = 0
        pred_batch[padding] = 0

        # human model param l2 loss
        human_model_loss = F.mse_loss(pred_batch, input_batch, reduction="none")

        # KL divergence loss
        mu, logvar = batch["mu"], batch["sigma"]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # vertex loss
        # initialize SMPL model
        smpl_model = SMPLModel()
        # get the vertex locations
        pred_vertex_locs = smpl_model(pred_batch)
        gt_vertex_locs = smpl_model(input_batch)
        # compute the vertex loss
        vertex_locs = F.mse_loss(pred_vertex_locs, gt_vertex_locs, reduction="none")

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
            loss_weight["vertex_loss"] * vertex_locs
        
        return total_loss.mean()
