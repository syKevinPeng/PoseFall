
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Encoder
from .decoder_with_init_pose import DecodeWithInitPose
from .loss import compute_in_phase_loss, compute_init_pose_loss
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
                        latent_dim=self.latent_dim*2, 
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

        batch[f"{phase}_combined_poses"] = batch[f"{phase}_combined_poses"].to(self.device)
        batch[f"{phase}_label"] = batch[f"{phase}_label"].to(self.device)
        batch[f"{phase}_src_key_padding_mask"] = batch[f"{phase}_src_key_padding_mask"].to(self.device)
        prev_phase = self.phase_names[self.phase_names.index(phase)-1]
        # add initial pose to the input batch
        if phase == self.phase_names[0]:
            batch[f'{phase}_init_pose'] = torch.zeros_like(batch[f"{phase}_combined_poses"][:, 0, :]).to(self.device)
        else:
            batch[f'{phase}_init_pose'] = batch[f"{prev_phase}_combined_poses"][:, -1, :].to(self.device)
            print(f'prev_phase: {prev_phase}, phase: {phase}')
        return batch
    
    def forward(self, batch):
        # batch = self.prepare_batch(batch)
        for phase in self.phase_names:
            batch = self.prepare_batch(batch, phase)
            # Encoder
            encoder_output = getattr(self, f"{phase}_encoder")(batch)
            batch.update(encoder_output)
            # reparameterize
            batch[f"{phase}_z"] = self.reparameterize(batch, phase)
            # Decoder
            batch.update(getattr(self, f"{phase}_decoder")(batch))
        return batch

    
    def compute_loss(self, batch):
        """
        loss function for 3E3D RNN model
        in_phase_loss: human model param l2 loss + KL loss +  vertex loss
        init pose loss: l2 loss between the initial pose and the first frame pose
        """
        loss_dict = {}
        losses = []
        for phase in self.phase_names:
            in_phase_loss, loss_dict_1 = compute_in_phase_loss(batch, phase_name = phase, weight_dict=self.config['loss_config'])
            if phase == self.phase_names[0]:
                init_pose_loss, loss_dict_2 = 0, {}
            else:
                init_pose_loss, loss_dict_2 = compute_init_pose_loss(batch, phase_name = phase, weight_dict=self.config['loss_config'])
            total_loss = in_phase_loss + init_pose_loss
            loss_dict.update(loss_dict_1)
            loss_dict.update(loss_dict_2)
            losses.append(total_loss)
            loss_dict[f"{phase}_total_loss"] = total_loss.item()
        return losses, loss_dict
    

    def generate(self, batch):
        """
        generate the output from the model. Different from 3E3D, we need to generate sequentially
        """
        batch_size = batch[f"{self.phase_names[0]}_label"].size(0)
        model_input_batch = {}
        for phase in self.phase_names:
            # sample noise
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            model_input_batch[f"{phase}_z"] = z
            model_input_batch[f"{phase}_label"] = batch[f"{phase}_label"]
            model_input_batch[f"{phase}_src_key_padding_mask"] = batch[f"{phase}_mask"]
            # add initial pose to the input batch
            if phase == self.phase_names[0]:
                model_input_batch[f'{phase}_init_pose'] = torch.zeros(batch_size, 1, self.num_joints, self.feat_dim).to(self.device)
            else:
                last_phase_name = self.phase_names[self.phase_names.index(phase)-1]
                model_input_batch[f'{phase}_init_pose'] = model_input_batch[f"{last_phase_name}_output"][:, -1, :].to(self.device)

            model_input_batch.update(getattr(self, f"{phase}_decoder")(model_input_batch))
        return model_input_batch
        

        