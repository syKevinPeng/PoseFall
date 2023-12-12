from smplx import SMPLLayer
import torch.nn as nn
from data_processing.utils import parse_output
from icecream import ic
from data_processing.utils import rotation_6d_to_matrix
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SMPLModel(nn.Module):
    """
    Using SMPL layer to generate vertex locations for the human body
    """
    def __init__(self, SMPL_model_path = "/home/siyuan/research/PoseFall/data/SMPL_cleaned/SMPL_FEMALE.pkl") -> None:
        super().__init__()
        # initialize SMPL model
        self.human_model = SMPLLayer(
            model_path = SMPL_model_path,
            creat_body_pose=True,
        ).to(DEVICE)

    def forward(self, batch):
        """
        we discard the locationa and rotation of the armature. 
        Only calculate the vertex transformation resulted from joint rotation
        """
        parsed_output = parse_output(batch)
        bone_rot = parsed_output["bone_rot"]
        batch_size = bone_rot.size(0)
        # convert continuous rotation representationt to rotation matrix
        bone_rot = rotation_6d_to_matrix(bone_rot)
        # body pose shape requirement: BxJx3x3
        body_pose = bone_rot.reshape(-1, 24, 3, 3)[:, 1:, :, :]
        SMPL_output = self.human_model(body_pose = body_pose, return_verts=True)
        verts = SMPL_output.vertices
        verts = verts.reshape(batch_size, -1, verts.size(-2), verts.size(-1))
        return verts

def human_param_loss(pred_batch, input_batch):
    """
    human model param l2 loss. Note that this contains the armature rotation, translation as well as the 
    bone rotation.
    """
    return F.mse_loss(pred_batch, input_batch, reduction="mean")

def kl_divergence(mu, logvar):
    """
    compute the KL divergence between the learned distribution and the standard normal distribution
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def vertex_loss(pred_batch, input_batch):
    """
    vertex loss: we ignore the translation and rotation of the armature. Only calculate the vertex transformation
    """
    # initialize SMPL model
    smpl_model = SMPLModel().eval().to(DEVICE)
    # get the vertex locations
    pred_vertex_locs = smpl_model(pred_batch)
    gt_vertex_locs = smpl_model(input_batch)
    # compute the vertex loss
    vertex_locs = F.mse_loss(pred_vertex_locs, gt_vertex_locs, reduction="mean")
    return vertex_locs