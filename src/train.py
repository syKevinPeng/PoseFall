import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import FallingData
from model.CVAE import CAVE
from icecream import ic
import wandb
from pathlib import Path
import datetime
from tqdm import tqdm

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data",
    help="path to the data directory",
)
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument(
    "--num_workers", type=int, default=4, help="number of workers for data loading"
)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument(
    "--wandb_project", type=str, default="posefall", help="wandb project name"
)
parser.add_argument("--wandb_mode", type=str, default="disabled", help="wandb mode")
parser.add_argument("--wandb_tag", type=str, default="train", help="wandb tag")
parser.add_argument(
    "--wandb_exp_name", type=str, default="test", help="wandb experiment name"
)
parser.add_argument(
    "--wandb_exp_description", type=str, default="", help="wandb experiment description"
)  # Added argument for experiment description
parser.add_argument(
    "--ckpt_path", type=str, default="", help="path to save checkpoints"
)
parser.add_argument(
    "--model_save_freq", type=int, default=200, help="frequency to save model"
)
args = parser.parse_args()

# ======================== prepare wandb ========================
# Initialize wandb
wandb.init(
    project=args.wandb_project,
    config=args,
    mode=args.wandb_mode,
    tags=args.wandb_tag,
    name=args.wandb_exp_name,
    notes=args.wandb_exp_description,
)

# ======================== prepare files/folders ========================
# check if ckpt path exists
if args.ckpt_path == "":
    ckpt_path = Path(wandb.run.dir)
else:
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print(f"Created checkpoint path {ckpt_path}")
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ckpt_path = ckpt_path / f"{current_time}_{args.wandb_exp_name}"
ckpt_path.mkdir(parents=True, exist_ok=True)

# ======================== define variables for training ========================
# TODO: create a config file for all the variables
# Define phases and data loader
PHASES = ["impa", "glit", "fall"]
data = FallingData(args.data_path)
dataloaders = torch.utils.data.DataLoader(
    data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):  # Epoch loop
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
    if (epoch + 1) % args.model_save_freq == 0:  # Save every 10 epochs
        checkpoint_path = ckpt_path / f"epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
