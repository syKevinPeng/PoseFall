import argparse
from json import decoder
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
train_config = args["train_config"]
data = FallingData(args["data_config"]["data_path"], data_aug=train_config["data_aug"])
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
if train_config["model_type"] == "CVAE":
    model = CVAE(num_classes_dict=num_class, config = args).to(DEVICE)
elif train_config["model_type"] == "CVAE_1D":
    model = CVAE1D(num_classes_dict=num_class, config = args).to(DEVICE)
else:
    raise ValueError(f"Model type {train_config['model_type']} not supported")
optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

# get model weight name
# model_weight_name = model.state_dict().keys()
# print(f"Model weight name: {model_weight_name}") #fall_decoder.seqTransDecoder

# load pretrained model
pretrained_weights = Path(args["data_config"]['pretrained_weights'])
if not pretrained_weights.exists():
    raise ValueError(f"Pretrained weights {pretrained_weights} does not exist")
weight = torch.load(pretrained_weights)

encoder_key = "encoder.seqTransEncoder"
decoder_key = "decoder.seqTransDecoder"

def load_weights(weight_key:str, pretrain_model_weight, my_model=model):
    # only get the encoder weights
    encoder_weight = {}
    for key in pretrain_model_weight.keys():
        if weight_key in key:
            encoder_weight[key] = pretrain_model_weight[key]

    duplicated_weight = {}
    for phase in PHASES:
        # update weights key name
        for key in encoder_weight.keys():
            if "Encoder" in weight_key:
                name = "encoder"
                new_key = re.sub(r"(encoder)(?=\.)", f"{phase}_{name}", key, count=1)
            elif "Decoder" in weight_key:
                name = "decoder"
                new_key = re.sub(r"(decoder)(?=\.)", f"{phase}_{name}", key, count=1)
                if new_key not in my_model.state_dict().keys():
                    raise Exception(f"Key {new_key} not in model state dict")
            else:
                raise ValueError(f"Weight key {weight_key} not supported")
            duplicated_weight[new_key] = encoder_weight[key]
    # check if the keys are the same
    for key in duplicated_weight.keys():
        if key not in my_model.state_dict().keys():
            print(f"Key {key} not in model state dict")
    # load the state dict to the model
    return duplicated_weight

pretrained_weights = {}
pretrained_weights.update(load_weights(encoder_key, weight))
pretrained_weights.update(load_weights(decoder_key, weight))
model.load_state_dict(pretrained_weights, strict=False)

for epoch in range(train_config['epochs']):  # Epoch loop
    epoch_loss = 0
    print(f"=== training on epoch {epoch} ===")
    for i_batch, (data_dict) in tqdm(enumerate(dataloaders), total=len(dataloaders)):
        optimizer.zero_grad()
        batch = model(batch=data_dict)
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    wandb.log({"epoch_loss": epoch_loss})
    print(f"Epoch {epoch}: loss {epoch_loss}")
    # Save model checkpoint
    if (epoch + 1) % train_config['model_save_freq'] == 0:  # Save every 10 epochs
        checkpoint_path = ckpt_path / f"epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
