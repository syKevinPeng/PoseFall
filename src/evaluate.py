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
import argparse
from visulization.pose_vis import visulize_poses
import imageio
# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHASES = ["impa", "glit", "fall"]

# ckpt_path = "/home/siyuan/research/PoseFall/src/wandb/run-20231221_215107-4o2q6lod/files/2023-12-21_21-51-07_exp0/epoch_1999.p5"

def load_ckpts(ckpt_path):
    """
    Load the checkpoint from the path
    """

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint path {ckpt_path} does not exist")
    # load checkpoint
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    return state_dict


def prepare_data(data_path):
# ======================== prepare data ========================
# data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"Data path {data_path} does not exist")

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
    return dataloaders, num_class, impa_label, glit_label, fall_label


if __name__ == "__main__":
    parser = argparse.get_parser()
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to the checkpoint")
    parser.add_argument("--data_path", type=str, default="/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data", help="Path to the data")
    parser.add_argument("--output_path", type=str, default="", help="Path to the output")
    parser.add_argument("--num_to_gen", type=int, default=1, help="Number of sequences to generate")
    parser.add_argument(
    "--wandb_project", type=str, default="posefall", help="wandb project name"
    )
    parser.add_argument("--wandb_mode", type=str, default="disabled", help="wandb mode")
    parser.add_argument("--wandb_tag", type=str, default="evaluate", help="wandb tag")
    parser.add_argument(
        "--wandb_exp_name", type=str, default="test", help="wandb experiment name"
    )
    parser.add_argument(
        "--wandb_exp_description", type=str, default="", help="wandb experiment description"
    )  # Added argument for experiment description

    args = parser.parse_args()

    state_dict = load_ckpts(args.ckpt_path)
    dataloaders, num_class, impa_label, glit_label, fall_label = prepare_data(args.data_path)
    if not Path(args.output_path).exists():
        print(f"Output path {args.output_path} does not exist... Creating it")
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
    # ======================== actual evaluation pipeline ========================
    # Initialize model and optimizer
    model = CAVE(phase_names=PHASES, num_classes_dict=num_class).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    for idx, data_dict in enumerate(tqdm(dataloaders)):
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
        frames = visulize_poses(df)
        # save frames
        imageio.mimsave(Path(args.output_path)/f"{idx}_sequences.gif", frames, fps=30)
        if idx == args.num_to_gen:
            break

    print(f"Generated {args.num_to_gen} sequences and saved them to {args.output_path}")
