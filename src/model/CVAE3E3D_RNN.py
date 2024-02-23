
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Encoder
from .decoder_with_init_pose import DecodeWithInitPose

class CVAE3E3D_RNN(nn.Module):
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
                DecodeWithInitPose(num_classes=data_config_dict[phase]["label_size"], 
                        phase_names=phase, 
                        latent_dim=self.latent_dim, 
                        njoints=data_config_dict["num_joints"], 
                        nfeats=data_config_dict["feat_dim"],
                        input_feature_dim=self.num_joints*self.feat_dim)
            )
        self.config = config
        print(f"CAVE model initialized with phases: {self.phase_names}")

    
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
    
    def prepare_batch(self, batch, phase):
        input_batch = {}
        input_batch[f"{phase}_combined_poses"] = batch[f"{phase}_combined_poses"].to(self.device)
        input_batch[f"{phase}_label"] = batch[f"{phase}_label"].to(self.device)
        input_batch[f"{phase}_src_key_padding_mask"] = batch[f"{phase}_src_key_padding_mask"].to(self.device)
        # add initial pose to the input batch
        input_batch[f'{phase}_init_pose'] = batch[f"{phase}_combined_poses"][:, 0, :].unsqueeze(1)
        print(f'input_batch[f"{phase}_init_pose"] shape: {input_batch[f"{phase}_init_pose"].shape}')
        print(f'combined_poses shape: {batch[f"{phase}_combined_poses"].shape}')
        return input_batch
    
    def forward(self, batch):
        # batch = self.prepare_batch(batch)
        for phase in self.phase_names:
            input_batch = self.prepare_batch(batch, phase)
            # Encoder
            encoder_output = getattr(self, f"{phase}_encoder")(input_batch)
            input_batch.update(encoder_output)
            # reparameterize
            input_batch[f"{phase}_z"] = self.reparameterize(input_batch, phase)
            # Decoder
            input_batch.update(getattr(self, f"{phase}_decoder")(input_batch))


        