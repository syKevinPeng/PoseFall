import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from dataloader import FallingDataset3Phase, FallingDataset1Phase
from model.CVAE3E3D import CVAE3E3D
from model.CVAE3E1D import CVAE3E1D
from model.CVAE1E1D import CVAE1E1D
from icecream import ic
import pandas as pd
from data_processing.utils import (
    parse_output,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
)
from data_processing import joint_names
import argparse
from src.visulization.torch3d_vis import visulize_poses
import imageio
import yaml, wandb

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_model(model_type, num_class):
    """
    Get the model based on the config
    """
    if model_type == "CVAE3E3D":
        model = CVAE3E3D(num_classes_dict=num_class, config=args).to(DEVICE)
    elif model_type == "CVAE3E1D":
        model = CVAE3E1D(data_config_dict=num_class, config=args).to(DEVICE)
    elif model_type == "CVAE1E1D":
        model = CVAE1E1D(num_classes=num_class, config=args).to(DEVICE)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    return model

def prepare_input_instance(data_instance):
    if "combined_label" in data_instance.keys():
        num_class = {"combined": data_instance["combined_label"].size(0)}
        print(f"Number of classes: {num_class}")
        num_frames, num_joints, feat_dim = data_instance["combined_combined_poses"].size()
        num_class.update(
            {"num_frames": num_frames, "num_joints": num_joints, "feat_dim": feat_dim}
        )
        input_type = "combined"
    else:
        # Get number of classes for each phase
        impa_label = data_instance["impa_label"]
        glit_label = data_instance["glit_label"]
        fall_label = data_instance["fall_label"]
        num_class = {
            "impa": impa_label.size(0),
            "glit": glit_label.size(0),
            "fall": fall_label.size(0),
        }
        input_type  = "seperated"

    return num_class, input_type

def get_model_and_dataset(args):
    model_name = args["generate_config"]["model_type"]
    train_config = args["generate_config"]
    # ======================== actual training pipeline ========================
    # Initialize model and optimizer
    if model_name== "CVAE3E3D":
        dataset = FallingDataset3Phase(
        args["data_config"]["data_path"], data_aug=False, max_frame_dict=args["constant"]["max_frame_dict"], phase=PHASES
        )
        data_configs = {}
        for phase in PHASES:
            num_frames, num_joints, feat_dim = dataset[0][f"{phase}_combined_poses"].size()
            data_configs.update({
                phase:{"num_frames": num_frames, "label_size":dataset[0][f"{phase}_label"].size(0)}
            })
        data_configs.update({
            "num_joints": num_joints, "feat_dim": feat_dim, 
        })
        model = CVAE3E3D(data_config_dict=data_configs, config=args).to(DEVICE)
        input_type = "seperated"
    elif model_name== "CVAE3E1D":
        dataset = FallingDataset3Phase(
        args["data_config"]["data_path"], data_aug=False, max_frame_dict=args["constant"]["max_frame_dict"]
        )
        data_configs = {}
        for phase in PHASES:
            num_frames, num_joints, feat_dim = dataset[0][f"{phase}_combined_poses"].size()
            data_configs.update({
                phase:{"num_frames": num_frames, "label_size":dataset[0][f"{phase}_label"].size(0)}
            })
        data_configs.update({
            "num_joints": num_joints, "feat_dim": feat_dim, 
        })
        print(f'Data Configs: \n {data_configs}')
        model = CVAE3E1D(data_config_dict=data_configs, config=args).to(DEVICE)
        input_type = "seperated"
    elif model_name== "CVAE1E1D":
        dataset = FallingDataset1Phase(
        args["data_config"]["data_path"], data_aug=False, max_frame_dict=args["constant"]["max_frame_dict"]
        )
        data_configs = {"combined": dataset[0]["combined_label"].size(0)}
        print(f"Number of classes: {data_configs}")
        num_frames, num_joints, feat_dim = dataset[0]["combined_combined_poses"].size()
        print(f"Input size: {num_frames, num_joints, feat_dim}")
        data_configs.update(
            {"num_frames": num_frames, "num_joints": num_joints, "feat_dim": feat_dim}
        )
        model = CVAE1E1D(num_classes=data_configs, config=args).to(DEVICE)
        # data = FallingDataset(args["data_config"]["data_path"], data_aug=train_config["data_aug"])
        input_type = "combined"
    else:
        raise ValueError(f"Model type {train_config['model_type']} not supported")
    return model, dataset, input_type



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
    wandb_config = args["generate_config"]["wandb_config"]
    wandb.init(
        project=wandb_config["wandb_project"],
        config=args,
        mode=wandb_config["wandb_mode"],
        tags=wandb_config["wandb_tags"],
        name=wandb_config["wandb_exp_name"],
        notes=wandb_config["wandb_description"],
    )
    PHASES = args["constant"]["PHASES"]
    generate_config = args["generate_config"]
    state_dict = load_ckpts(generate_config["ckpt_path"])
    # ======================== prepare data ========================
    data_path = Path(generate_config["data_path"])
    if not data_path.exists():
        raise ValueError(f"Data path {data_path} does not exist")

    # dataset = FallingDataset1Phase(data_path, max_frame_dict=args["constant"]["max_frame_dict"])
    if not Path(generate_config["output_path"]).exists():
        print(f"Output path {generate_config['output_path']} does not exist... Creating it")
        Path(generate_config["output_path"]).mkdir(parents=True, exist_ok=True)
    # ======================== prepare model ========================
    # num_class, input_type=prepare_input_instance(dataset[0])
    model, dataset, input_type = get_model_and_dataset(args)
    dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    # model = get_model(generate_config["model_type"], num_class=num_class)
    model.load_state_dict(state_dict)
    model.eval()

    for idx, data_dict in enumerate(tqdm(dataloaders)):
        if input_type == "seperated":
            input_batch = {
                "impa_label": data_dict["impa_label"].to(DEVICE),
                "glit_label": data_dict["glit_label"].to(DEVICE),
                "fall_label": data_dict["fall_label"].to(DEVICE),
                "impa_mask": data_dict["impa_src_key_padding_mask"].to(DEVICE),
                "glit_mask": data_dict["glit_src_key_padding_mask"].to(DEVICE),
                "fall_mask": data_dict["fall_src_key_padding_mask"].to(DEVICE),
            }
            batch_size = input_batch["impa_label"].size(0)
        elif input_type == "combined":
            input_batch = {
                "combined_label": data_dict["combined_label"].to(DEVICE),
                "combined_mask": data_dict["combined_src_key_padding_mask"].to(DEVICE),
            }
            batch_size = input_batch["combined_label"].size(0)
        
        genreated_batch = model.generate(input_batch)
        # genreated_batch["output"] = data_dict["combined_combined_poses"].reshape(1, 430, 156).to(DEVICE)
        if generate_config["model_type"] == "CVAE3E3D":
            whole_sequences = []
            for phase in PHASES:
                model_output = genreated_batch[f"{phase}_output"]
                model_output = model_output
                batch_size = model_output.size(0)
                # remove padding
                model_output = model_output[~(input_batch[f"{phase}_mask"].bool())]
                num_fram, num_joints, feat_dim = model_output.size()
                phase_output = model_output.reshape(batch_size, -1, num_joints, feat_dim).cpu().detach()
                whole_sequences.append(phase_output)
            whole_sequences = torch.concat(whole_sequences, axis=1)
        elif generate_config["model_type"] == "CVAE3E1D" or generate_config["model_type"] == "CVAE1E1D":
            whole_sequences = genreated_batch["combined_output"]
            whole_sequences = whole_sequences
            # remove padding
            batch_size, num_fram, num_joints, feat_dim = whole_sequences.size()
            whole_sequences = whole_sequences[~(input_batch["combined_mask"].bool())]
            whole_sequences = whole_sequences.reshape(batch_size, -1, num_joints, feat_dim).cpu().detach()


        # parse the output
        parsed_seequnces = parse_output(whole_sequences)
        # convert the rotation to rotation matrix
        bone_rot = rotation_6d_to_matrix(parsed_seequnces["bone_rot"])
        # convert the rotation matrix to euler angle
        bone_rot = matrix_to_euler_angles(bone_rot, "XYZ")
        # flatten the bone rot
        bone_rot = bone_rot.reshape(bone_rot.size(0), bone_rot.size(1), -1)
        arm_rot = rotation_6d_to_matrix(parsed_seequnces["arm_rot"])
        arm_rot = matrix_to_euler_angles(arm_rot, "XYZ")
        arm_loc = parsed_seequnces["arm_loc"]
        # form the output as a dataframe
        joint_name = joint_names.SMPL_JOINT_NAMES
        joint_name = np.array(
            [[f"{name}_x", f"{name}_y", f"{name}_z"] for name in joint_name]
        ).flatten()
        # get the first item in the batch
        data = torch.concatenate([arm_rot, arm_loc, bone_rot], axis=2)[0]
        col_names = ["arm_rot_x", "arm_rot_y", "arm_rot_z"] + ["arm_loc_x", "arm_loc_y", "arm_loc_z"]+ list(joint_name)
        df = pd.DataFrame(
            data=data,
            columns= col_names,
        )

        # save the dataframe
        df.to_csv(Path(generate_config['output_path']) / f"{idx}_sequences.csv")
        # visulization
        frames = visulize_poses(df)
        # save frames
        imageio.mimsave(Path(generate_config['output_path']) / f"{idx}_sequences.gif", frames, fps=30)
        if idx == generate_config["num_to_gen"] - 1:
            break

    print(f"Generated {generate_config['num_to_gen']} sequences and saved them to {generate_config['output_path']}")
