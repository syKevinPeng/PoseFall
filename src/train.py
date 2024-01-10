import argparse
from distutils.ccompiler import new_compiler
from json import decoder
from typing import cast
from click import option
import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
from dataloader import FallingDataset3Phase, FallingDataset1Phase
from model.CVAE3E3D import CVAE3E3D
from model.CVAE3E1D import CVAE3E1D
from model.CVAE1E1D import CVAE1E1D
from icecream import ic
import wandb, yaml
from pathlib import Path
import datetime, re
from tqdm import tqdm
import numpy as np

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default=Path(__file__).parent.joinpath("config.yaml"),
        help="path to config file",
    )
    cmd_args = parser.parse_args()
    # load config file
    with open(cmd_args.config_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args

def prepare_wandb():
    # ======================== prepare wandb ========================
    # Initialize wandb
    wandb_config = args["wandb_config"]
    wandb.init(
        project=wandb_config["wandb_project"],
        config=args,
        mode=wandb_config["wandb_mode"],
        tags=wandb_config["wandb_tags"],
        name=wandb_config["wandb_exp_name"],
        notes=wandb_config["wandb_description"],
    )
    return wandb_config

def get_model_and_dataset(args):
    model_name = args["train_config"]["model_type"]
    train_config = args["train_config"]
    # ======================== actual training pipeline ========================
    # Initialize model and optimizer
    if model_name== "CVAE3E3D":
        dataset = FallingDataset3Phase(
        args["data_config"]["data_path"], data_aug=train_config["data_aug"], max_frame_dict=args["constant"]["max_frame_dict"], phase=PHASES
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

    elif model_name== "CVAE3E1D":
        dataset = FallingDataset3Phase(
        args["data_config"]["data_path"], data_aug=train_config["data_aug"], max_frame_dict=args["constant"]["max_frame_dict"]
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

    elif model_name== "CVAE1E1D":
        dataset = FallingDataset1Phase(
        args["data_config"]["data_path"], data_aug=train_config["data_aug"], max_frame_dict=args["constant"]["max_frame_dict"]
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
    else:
        raise ValueError(f"Model type {train_config['model_type']} not supported")
    return model, dataset

def prepare_pretrain_weights(args, curr_model_weights):
    # load pretrained model
    pretrained_weights = Path(args["data_config"]["pretrained_weights"])
    model_name = args["train_config"]["model_type"]
    if not pretrained_weights.exists():
        raise ValueError(f"Pretrained weights {pretrained_weights} does not exist")
    pretrain_loaded_weights = torch.load(pretrained_weights)
    weights_to_skip = ["encoder.muQuery", "encoder.sigmaQuery", "decoder.actionBiases","encoder.skelEmbedding.weight","decoder.finallayer.weight","decoder.finallayer.bias"]
    pretrain_loaded_weights = {
        k: v for k, v in pretrain_loaded_weights.items() if k not in weights_to_skip
    }

    if model_name == "CVAE3E1D":
        new_pretrain_laoded_weights = {}
        # copy encoder weights:
        for phase in PHASES:
            for key in pretrain_loaded_weights.keys():
                if "encoder" in key:
                    new_key = re.sub(r"(encoder)(?=\.)", f"{phase}_encoder", key, count=1)
                    new_pretrain_laoded_weights[new_key] = pretrain_loaded_weights[key]
                else:
                    new_pretrain_laoded_weights[key] = pretrain_loaded_weights[key]
        pretrain_loaded_weights = new_pretrain_laoded_weights
    
    # print out weights that are not loaded
    for key in curr_model_weights.keys():
        if key not in pretrain_loaded_weights.keys():
            print(f"Key {key} is not loaded")
    return pretrain_loaded_weights

if __name__ == "__main__":
    
    args = parse_args()
    wandb_config = prepare_wandb()
    train_config = args["train_config"]
    PHASES = args["constant"]["PHASES"]

    # ======================== prepare files/folders ========================
    # check if ckpt path exists
    if args["data_config"]["ckpt_path"] == "":
        ckpt_path = Path(wandb.run.dir)
    else:
        ckpt_path = Path(args["data_config"]["ckpt_path"])
        if not ckpt_path.exists():
            ckpt_path.mkdir(parents=True, exist_ok=True)
            print(f"Created checkpoint path {ckpt_path}")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_path = ckpt_path / f"{current_time}_{wandb_config['wandb_exp_name']}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    # ======================== define variables for training ========================
    # get model, dataset, dataloader and optimizer
    model, dataset = get_model_and_dataset(args=args)
    curr_model_weights = model.state_dict()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"])
    dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
    )

    # ======================== load pretrained weights ========================
    loaded_weights = prepare_pretrain_weights(args, curr_model_weights)
    model.load_state_dict(loaded_weights, strict=False)

    # ======================== actual training pipeline ========================
    for epoch in range(train_config["epochs"]):  # Epoch loop
        epoch_loss = 0
        cum_loss_dict = {}
        print(f"=== training on epoch {epoch} ===")
        for i_batch, (data_dict) in tqdm(
            enumerate(dataloaders), total=len(dataloaders)
        ):
            optimizer.zero_grad()
            batch = model(batch=data_dict)
            loss, loss_dict = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # cumulate the loss in the loss dict
            for key in loss_dict.keys():
                if key not in cum_loss_dict.keys():
                    cum_loss_dict[key] = loss_dict[key]
                else:
                    cum_loss_dict[key] += loss_dict[key]

        wandb.log({"epoch_loss": cum_loss_dict})
        # print all loss in the loss dict in a line
        print(f"epoch {epoch} loss: {cum_loss_dict}")
        # Save model checkpoint
        if (epoch+1) % train_config["model_save_freq"] == 0:  
            checkpoint_path = ckpt_path / f"epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
