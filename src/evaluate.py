import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from dataloader import FallingData
from model.CVAE import CAVE
from icecream import ic
from data_processing.utils import parse_output
# Set device

# ======================== prepare ckpt ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint path
ckpt_path = "/home/siyuan/research/PoseFall/src/wandb/run-20231214_025139-ijy4tddr/files/2023-12-14_02-51-40_exp0"
ckpt_path = Path(ckpt_path)
if not ckpt_path.exists():
    raise ValueError(f"Checkpoint path {ckpt_path} does not exist")
# get all file under the ckpt path and sort them, get the latest one
ckpt_list = list(ckpt_path.glob("*.p5"))
ckpt_list.sort()
ckpt_file_path = ckpt_list[-1]
# load checkpoint
state_dict = torch.load(ckpt_file_path, map_location=DEVICE)

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
    batch_size=4,
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
            "impa_length": data_dict["impa_lengths"].to(DEVICE),
            "glit_length": data_dict["glit_lengths"].to(DEVICE),
            "fall_length": data_dict["fall_lengths"].to(DEVICE),

    }
    genreated_batch = model.generate(input_batch)
    whole_sequences = []
    for phase in PHASES:
        model_output = genreated_batch[f"{phase}_output"]
        model_output = model_output.cpu().detach().numpy()
        whole_sequences.append(model_output)
    whole_sequences = np.concatenate(whole_sequences, axis=1)
    ic(whole_sequences.shape)
    break
