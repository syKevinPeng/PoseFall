from smplx import SMPLLayer
import torch.nn as nn
from data_processing.utils import parse_output
from icecream import ic
from data_processing.utils import rotation_6d_to_matrix
import torch
import torch.nn.functional as F
import scipy

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
        joint_locs = SMPL_output.joints
        verts = verts.reshape(batch_size, -1, verts.size(-2), verts.size(-1))
        joint_locs = joint_locs.reshape(batch_size, -1, joint_locs.size(-2), joint_locs.size(-1))
        return verts, joint_locs

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
    pred_vertex_locs, _ = smpl_model(pred_batch)
    gt_vertex_locs,_ = smpl_model(input_batch)
    # compute the vertex loss
    vertex_locs = F.mse_loss(pred_vertex_locs, gt_vertex_locs, reduction="mean")
    return vertex_locs

def compute_in_phase_loss(batch, phase_name):
    if phase_name:
        pred_batch = batch[f"{phase_name}_output"]
        input_batch = batch[f"{phase_name}_combined_poses"]
        mask_batch = batch[f"{phase_name}_src_key_padding_mask"]
        mu, logvar = batch[f"{phase_name}_mu"], batch[f"{phase_name}_sigma"]
    else:
        pred_batch = batch["output"]
        # input_batch = batch["combined_poses"]
        mask_batch = batch["src_key_padding_mask"]
        mu, logvar = batch["mu"], batch["sigma"]
        for phase in PHASES:
            input_batch = batch[f"{phase}_combined_poses"]
            print(f"input batch shape: {input_batch.size()}")
            break

    padding = ~(mask_batch.bool().unsqueeze(-1).expand(-1, -1, pred_batch.size(-1)))
    pred_batch = pred_batch * padding
    input_batch = input_batch * padding

    # human model param l2 loss
    human_model_loss = human_param_loss(pred_batch, input_batch)

    # KL divergence loss
    kl_loss = kl_divergence(mu, logvar)

    # vertex loss
    vertex_locs_loss = vertex_loss(pred_batch, input_batch)
    # loss weight
    loss_weight = {
        "human_model_loss": 1,
        "kl_loss": 0.1,
        "vertex_loss": 1,
    }
    # compute loss
    total_phase_loss = (
        loss_weight["human_model_loss"] * human_model_loss
        + loss_weight["kl_loss"] * kl_loss
        + loss_weight["vertex_loss"] * vertex_locs_loss
    )
    return total_phase_loss

def compute_inter_phase_loss(phase_names,batch):
    """
    take 10% of the end of the first phase and 10% of the beginning of the second phase
    compute the first derivative of the joint location        
    """
    inter_phase_loss = 0
    loss_weight = {
        "var_loss": 0.1,
        "loc_loss": 1,
    }
    for i in range(len(phase_names)-1):
        first_phase = phase_names[i]
        second_phase = phase_names[i+1]
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
        inter_phase_loss += torch.mean(joint_locs_diff_var)*loss_weight["var_loss"]

        # Spline loss: calculate the different between the actual location and the interpolated location
        interpolated_joint_locs = scipy.interpolate.CubicSpline(
            x=torch.linspace(0, 1, joint_locs.size(1)), y=joint_locs.cpu().detach().numpy(), axis=1
        )(torch.linspace(0, 1, joint_locs.size(1)))
        interpolated_joint_locs = torch.tensor(interpolated_joint_locs).to(DEVICE)
        
        # calculate the l2 difference between the actual location and the spline location
        spline_loss = torch.mean(torch.square(joint_locs - interpolated_joint_locs))
        inter_phase_loss += spline_loss*loss_weight["loc_loss"]

    return inter_phase_loss