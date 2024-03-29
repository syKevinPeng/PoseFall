from smplx import SMPLLayer
import torch.nn as nn
from ..data_processing.utils import rotation_6d_to_matrix
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

    def forward(self, batch, rotation_and_transl = False):
        """
        we discard the locationa and rotation of the armature. 
        Only calculate the vertex transformation resulted from joint rotation
        """
        # parsed_output = parse_output(batch)
        # bone_rot = parsed_output["bone_rot"]
        # check the shape of the rotation and translation
        if rotation_and_transl:
            translation = batch[:, 24, :3]
            rotation = batch[:, 25, :]
            rotation = rotation_6d_to_matrix(rotation)
            batch = batch[:, :24, :]
        else:
            rotation = translation = None
        if batch.size(-2) != 24:
            raise ValueError(f"The poses should have 24 joints, but the have the shape of {batch.size()}")
        batch_size = batch.size(0)
        bone_rot = batch
        # convert continuous rotation representationt to rotation matrix
        bone_rot = rotation_6d_to_matrix(bone_rot)
        # body pose shape requirement: BxJx3x3
        body_pose = bone_rot.reshape(-1, 24, 3, 3)[:, 1:, :, :]
        SMPL_output = self.human_model(body_pose = body_pose, return_verts=True, transl = translation, global_orient = rotation)
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

def vertex_loss(pred_batch, input_batch, with_trans_rot = False):
    """
    vertex loss: we ignore the translation and rotation of the armature. Only calculate the vertex transformation
    """
    # initialize SMPL model
    smpl_model = SMPLModel().eval().to(DEVICE)
    # get the vertex locations
    pred_vertex_locs, _ = smpl_model(pred_batch, rotation_and_transl = with_trans_rot )
    gt_vertex_locs,_ = smpl_model(input_batch, rotation_and_transl = with_trans_rot)
    # compute the vertex loss
    vertex_locs = F.mse_loss(pred_vertex_locs, gt_vertex_locs, reduction="mean")
    return vertex_locs

def compute_in_phase_loss(batch, phase_name, weight_dict):
    # if phase_name: = 
    pred_batch = batch[f"{phase_name}_output"]
    input_batch = batch[f"{phase_name}_combined_poses"]
    mask_batch = batch[f"{phase_name}_src_key_padding_mask"].bool()
    mu, logvar = batch[f"{phase_name}_mu"], batch[f"{phase_name}_sigma"]

    batch_size = input_batch.size(0)
    pred_batch = pred_batch[~mask_batch]
    input_batch = input_batch[~mask_batch]

    # human model param l2 loss
    human_model_loss = human_param_loss(pred_batch, input_batch)

    # KL divergence loss
    kl_loss = kl_divergence(mu, logvar)

    # get the bone_rot from the input batch
    all_fram_num, num_joints, feat_dim = input_batch.size()
    bone_rot_input_batch = input_batch[:, :24, :]
    bone_rot_pred_batch = pred_batch[:, :24, :]
    # vertex loss
    vertex_locs_loss = vertex_loss(bone_rot_pred_batch, bone_rot_input_batch)
    # loss weight
    loss_weight = weight_dict
    # compute loss
    total_phase_loss = (
        loss_weight["human_model_loss_weight"] * human_model_loss
        + loss_weight["kl_loss_weight"] * kl_loss
        + loss_weight["vertex_loss_weight"] * vertex_locs_loss
    )
    loss_dict = {
        f"{phase_name}_human_model_loss": (loss_weight["human_model_loss_weight"] * human_model_loss).item(),
        f"{phase_name}_vertex_loss": (loss_weight["vertex_loss_weight"] * vertex_locs_loss).item(),
        f"{phase_name}_kl_loss": (loss_weight["kl_loss_weight"] * kl_loss).item(),
        f"{phase_name}_total_loss": (total_phase_loss).item(),
     }
    return total_phase_loss, loss_dict

def compute_inter_phase_loss(phase_names,batch, loss_weights_dict):
    """
    take 10% of the end of the first phase and 10% of the beginning of the second phase
    compute the first derivative of the joint location        
    """

    inter_phase_loss = 0
    loss_dict = {}
    for i in range(len(phase_names)-1):
        first_phase = phase_names[i]
        second_phase = phase_names[i+1]
        pred_first_phase = batch[f"{first_phase}_output"] # shape: (batch_size, num_frames, num_joints, feat_dim)
        pred_second_phase = batch[f"{second_phase}_output"]
        first_phase_last_frame = pred_first_phase[:, -1, :]
        second_phase_first_frame = pred_second_phase[:, 0, :] # shape: (batch_size, num_joints, feat_dim)
        recon_loss = vertex_loss(first_phase_last_frame[:, :26, :], second_phase_first_frame[:, :26, :], with_trans_rot=True)*loss_weights_dict["inter_phase_loss_weight"]
        loss_dict.update({
            f"{first_phase}_{second_phase}_recon_loss": recon_loss.item()*loss_weights_dict["inter_phase_loss_weight"],
        })

    return inter_phase_loss, loss_dict

def compute_init_pose_loss(batch, phase_name, weight_dict):
    """
    compute the l2 loss between the initial pose and the first frame pose
    """
    gt_init_pose = batch[f'{phase_name}_init_pose']
    pred_init_pose = batch[f'{phase_name}_output'][:, 0, :]
    param_loss = F.mse_loss(gt_init_pose,pred_init_pose)*weight_dict["init_pose_param_loss_weight"]
    # reconstruct the initial pose
    recon_loss = vertex_loss(gt_init_pose[:, :26, :], pred_init_pose[:, :26, :], with_trans_rot=True)*weight_dict["init_pose_vertex_loss_weight"]
    loss = param_loss + recon_loss #+ human_center_loss
    loss_dict = {f"{phase_name}_init_pose_loss": param_loss.item(),
                                    f"{phase_name}_init_pose_recon_loss": recon_loss.item(),
                                    f"{phase_name}_total_init_pose_loss": loss.item()}
    return loss, loss_dict