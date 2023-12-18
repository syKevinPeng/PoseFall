import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from dataloader import FallingData
from model.CVAE import CAVE
from icecream import ic
import pandas as pd
from data_processing.utils import parse_output, rotation_6d_to_matrix, matrix_to_euler_angles
from data_processing import joint_names
from visulization.pose_vis import visulize_poses
# Set device

# ======================== prepare ckpt ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint path
ckpt_path = "/home/siyuan/research/PoseFall/src/wandb/latest-run/files/2023-12-14_02-51-40_exp0/epoch_1999.p5 "
ckpt_path = Path(ckpt_path)
if not ckpt_path.exists():
    raise ValueError(f"Checkpoint path {ckpt_path} does not exist")

# load checkpoint
state_dict = torch.load(ckpt_path, map_location=DEVICE)

# ======================== prepare data ========================
data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
data_path = Path(data_path)
if not data_path.exists():
    raise ValueError(f"Data path {data_path} does not exist")

# use the same training dataloader
# ======================== define variables for training ======================== 
# Define phases and data loader
PHASES = ["impa", "glit", "fall"]
data = FallingData(data_path)
dataloaders = torch.utils.data.DataLoader(
    data,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

# Get number of classes for each phase
impa_label = data[0]["impa_label"]
glit_label = data[0]["glit_label"]
fall_label = data[0]["fall_label"]
num_class = {
    "impa": impa_label.size(0),
    "glit": glit_label.size(0),
    "fall": fall_label.size(0),
}

# ======================== actual training pipeline ========================
# Initialize model and optimizer
model = CAVE(phase_names=PHASES, num_classes_dict=num_class).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

for data_dict in tqdm(dataloaders):
    input_batch = {
            "impa_label": data_dict["impa_label"].to(DEVICE),
            "glit_label": data_dict["glit_label"].to(DEVICE),
            "fall_label": data_dict["fall_label"].to(DEVICE),
            "impa_mask": data_dict["impa_src_key_padding_mask"].to(DEVICE),
            "glit_mask": data_dict["glit_src_key_padding_mask"].to(DEVICE),
            "fall_mask": data_dict["fall_src_key_padding_mask"].to(DEVICE),
    }
    genreated_batch = model.generate(input_batch)
    batch_size = input_batch["impa_label"].size(0)
    whole_sequences = []
    for phase in PHASES:
        model_output = genreated_batch[f"{phase}_output"]
        model_output = model_output.cpu().detach()
        # remove padding based on the mask
        mask = input_batch[f"{phase}_mask"]
        mask = mask.cpu().detach().bool()
        # remove the padding but keep the batch dimension
        model_output = model_output[mask, :].reshape(batch_size, -1, model_output.shape[-1])
        whole_sequences.append(model_output)
    whole_sequences = torch.concat(whole_sequences, axis=1)
    # parse the output
    parsed_seequnces = parse_output(whole_sequences)
    # convert the rotation to rotation matrix
    bone_rot = rotation_6d_to_matrix(parsed_seequnces["bone_rot"])
    # convert the rotation matrix to euler angle
    bone_rot = matrix_to_euler_angles(bone_rot, "XYZ")
    bone_rot = bone_rot.reshape(batch_size, bone_rot.size(1), -1)
    arm_rot = rotation_6d_to_matrix(parsed_seequnces["arm_rot"])
    arm_rot = matrix_to_euler_angles(arm_rot, "XYZ")
    arm_loc = parsed_seequnces["arm_loc"]
    # form the output as a dataframe
    joint_name = joint_names.SMPL_JOINT_NAMES
    joint_name = np.array([[f"{name}_x", f"{name}_y", f"{name}_z"] for name in joint_name]).flatten()
    # get the first item in the batch
    data = torch.concatenate([arm_loc, arm_rot, bone_rot], axis=2)[0]
    df = pd.DataFrame(data=data, 
                      columns=["arm_loc_x", "arm_loc_y", "arm_loc_z"] + ["arm_rot_x", "arm_rot_y", "arm_rot_z"] + list(joint_name))
    # visulization
    visulize_poses(df)
    break
