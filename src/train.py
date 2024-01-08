import argparse
from json import decoder
from typing import cast
from click import option
import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
from dataloader import FallingDataset, myFallingDataset
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
# Define phases and data loader
PHASES = args["constant"]["PHASES"]
train_config = args["train_config"]
# data = FallingDataset(args["data_config"]["data_path"], data_aug=train_config["data_aug"])
dataset = myFallingDataset(
    args["data_config"]["data_path"], data_aug=train_config["data_aug"], max_frame_dict=args["constant"]["max_frame_dict"]
)
dataloaders = torch.utils.data.DataLoader(
    dataset,
    batch_size=train_config["batch_size"],
    shuffle=False,
    num_workers=train_config["num_workers"],
)
if "combined_label" in dataset[0].keys():
    num_class = {"combined": dataset[0]["combined_label"].size(0)}
    print(f"Number of classes: {num_class}")
    num_frames, num_joints, feat_dim = dataset[0]["combined_combined_poses"].size()
    print(f"Input size: {num_frames, num_joints, feat_dim}")
    num_class.update(
        {"num_frames": num_frames, "num_joints": num_joints, "feat_dim": feat_dim}
    )
else:
    # Get number of classes for each phase
    impa_label = dataset[0]["impa_label"]
    glit_label = dataset[0]["glit_label"]
    fall_label = dataset[0]["fall_label"]
    num_class = {
        "impa": impa_label.size(0),
        "glit": glit_label.size(0),
        "fall": fall_label.size(0),
    }



def init():
    # ======================== actual training pipeline ========================
    # Initialize model and optimizer
    if train_config["model_type"] == "CVAE3E3D":
        model = CVAE3E3D(num_classes_dict=num_class, config=args).to(DEVICE)
    elif train_config["model_type"] == "CVAE3E1D":
        model = CVAE3E1D(num_classes_dict=num_class, config=args).to(DEVICE)
    elif train_config["model_type"] == "CVAE1E1D":
        model = CVAE1E1D(num_classes=num_class, config=args).to(DEVICE)
    else:
        raise ValueError(f"Model type {train_config['model_type']} not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    # get model weight name
    # model_weight_name = model.state_dict().keys()
    # print(f"Model weight name: {model_weight_name}") #fall_decoder.seqTransDecoder

    # load pretrained model
    pretrained_weights = Path(args["data_config"]["pretrained_weights"])
    if not pretrained_weights.exists():
        raise ValueError(f"Pretrained weights {pretrained_weights} does not exist")
    # loaded_weights = torch.load(pretrained_weights)
    loaded_weights_: "dict[str,torch.Torch]" = torch.load(
        "/home/siyuan/research/PoseFall/src/model/pretrained_models/uestc/checkpoint_1000.pth.tar"
    )

    weights_to_skip = ["encoder.muQuery", "encoder.sigmaQuery", "decoder.actionBiases","encoder.skelEmbedding.weight","decoder.finallayer.weight","decoder.finallayer.bias"]
    loaded_weights = {
        k: v for k, v in loaded_weights_.items() if k not in weights_to_skip
    }

    model.load_state_dict(loaded_weights, strict=False)
    return model, optimizer, loaded_weights_, loaded_weights


def main(model: nn.Module, optimizer: torch.optim.Optimizer):
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


if __name__ == "__main__":
    model, optimizer, loaded_weights_, loaded_weights = init()
    main( model, optimizer)
