import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from dataloader import FallingData
from model.CVAE import CVAE
from model.CVAE_1D import CVAE1D
from icecream import ic
import pandas as pd
from data_processing.utils import (
    parse_output,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
)
from data_processing import joint_names
import argparse
from visulization.pose_vis import visulize_poses
import imageio
import yaml, wandb

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config_path", type=str, default="config.yaml", help="path to config file"
    )
    cmd_args = parser.parse_args()
    # load config file
    with open(cmd_args.config_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # ======================== prepare wandb ========================
    # Initialize wandb
    wandb_config = args["eval_config"]["wandb_config"]
    wandb.init(
        project=wandb_config["wandb_project"],
        config=args,
        mode=wandb_config["wandb_mode"],
        tags=wandb_config["wandb_tags"],
        name=wandb_config["wandb_exp_name"],
        notes=wandb_config["wandb_description"],
    )
    PHASES = args["constant"]["PHASES"]
    eval_config = args["eval_config"]
    state_dict = load_ckpts(eval_config["ckpt_path"])
    dataloaders, num_class, impa_label, glit_label, fall_label = prepare_data(
        eval_config["data_path"]
    )
    if not Path(eval_config["output_path"]).exists():
        print(f"Output path {eval_config['output_path']} does not exist... Creating it")
        Path(eval_config["output_path"]).mkdir(parents=True, exist_ok=True)
    # ======================== actual evaluation pipeline ========================
    # Initialize model and optimizer
    if eval_config["model_type"] == "CVAE":
        model = CVAE(num_classes_dict=num_class, config = args).to(DEVICE)
    elif eval_config["model_type"] == "CVAE_1D":
        model = CVAE1D(num_classes_dict=num_class, config = args).to(DEVICE)
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
        if eval_config["model_type"] == "CVAE":
            whole_sequences = []
            for phase in PHASES:
                model_output = genreated_batch[f"{phase}_output"]
                model_output = model_output.cpu().detach()
                whole_sequences.append(model_output)
            whole_sequences = torch.concat(whole_sequences, axis=1)
        elif eval_config["model_type"] == "CVAE_1D":
            whole_sequences = genreated_batch["output"]
            whole_sequences = whole_sequences.cpu().detach()
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
        joint_name = np.array(
            [[f"{name}_x", f"{name}_y", f"{name}_z"] for name in joint_name]
        ).flatten()
        # get the first item in the batch
        data = torch.concatenate([arm_loc, arm_rot, bone_rot], axis=2)[0]
        df = pd.DataFrame(
            data=data,
            columns=["arm_loc_x", "arm_loc_y", "arm_loc_z"]
            + ["arm_rot_x", "arm_rot_y", "arm_rot_z"]
            + list(joint_name),
        )
        # save the dataframe
        df.to_csv(Path(eval_config['output_path']) / f"{idx}_sequences.csv")
        # visulization
        frames = visulize_poses(df)
        # save frames
        imageio.mimsave(Path(eval_config['output_path']) / f"{idx}_sequences.gif", frames, fps=30)
        if idx == eval_config["num_to_gen"] - 1:
            break

    print(f"Generated {eval_config['num_to_gen']} sequences and saved them to {eval_config['output_path']}")
