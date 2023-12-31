import argparse
from click import option
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import FallingData
from model.CVAE import CVAE
from model.CVAE_1D import CVAE1D
from icecream import ic
import wandb, yaml
from pathlib import Path
import datetime, re
from tqdm import tqdm

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--config_path", type=str, default="config.yaml", help="path to config file")
cmd_args = parser.parse_args()
# load config file
with open(cmd_args.config_path, "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
# ======================== prepare wandb ========================
# Initialize wandb
wandb_config = args['wandb_config']
wandb.init(
    project=wandb_config['wandb_project'],
    config=args,
    mode=wandb_config['wandb_mode'],
    tags=wandb_config['wandb_tags'],
    name=wandb_config['wandb_exp_name'],
    notes=wandb_config['wandb_description'],
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
train_config = args["train_hyperparameters"]
data = FallingData(args["data_config"]["data_path"])
dataloaders = torch.utils.data.DataLoader(
    data,
    batch_size=train_config["batch_size"],
    shuffle=True,
    num_workers=train_config["num_workers"],
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
if train_config["model_type"] == "CAVE":
    model = CVAE(phase_names=PHASES, num_classes_dict=num_class).to(DEVICE)
elif train_config["model_type"] == "CVAE_1D":
    model = CVAE1D(phase_names=PHASES, num_classes_dict=num_class).to(DEVICE)
else:
    raise ValueError(f"Model type {train_config['model_type']} not supported")
optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

# load pretrained model
pretrained_weights = Path("/home/siyuan/research/PoseFall/src/model/pretrained_models/humanact12/checkpoint_5000.pth.tar")
if not pretrained_weights.exists():
    raise ValueError(f"Pretrained weights {pretrained_weights} does not exist")
weight = torch.load(pretrained_weights)

# only get the encoder weights
encoder_weight = {}
for key in weight.keys():
    if "encoder.seqTransEncoder" in key:
        encoder_weight[key] = weight[key]
duplicated_weight = {}
for phase in PHASES:
    # update weights key name
    for key in encoder_weight.keys():
        new_key = re.sub(r"(encoder)(?=\.)", f"{phase}_encoder", key, count=1)
        duplicated_weight[new_key] = encoder_weight[key]
# check if the keys are the same
for key in duplicated_weight.keys():
    if key not in model.state_dict().keys():
        print(f"Key {key} not in model state dict")
# load the state dict to the model
model.load_state_dict(duplicated_weight, strict=False)

for epoch in range(train_config['epochs']):  # Epoch loop
    epoch_loss = 0
    print(f"=== training on epoch {epoch} ===")
    for i_batch, (data_dict) in tqdm(enumerate(dataloaders), total=len(dataloaders)):
        optimizer.zero_grad()
        batch = model(batch=data_dict)
        loss = model.compute_all_phase_loss(batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    wandb.log({"epoch_loss": epoch_loss})
    print(f"Epoch {epoch}: loss {epoch_loss}")
    # Save model checkpoint
    if (epoch + 1) % train_config['model_save_freq'] == 0:  # Save every 10 epochs
        checkpoint_path = ckpt_path / f"epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
