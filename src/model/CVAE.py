from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        return 
    
    def return_latent(self, batch):
        # encode
        batch.update(self.encoder(batch))
        # reparameterize
        batch["z"] = self.reparameterize(batch)
        return batch["z"]