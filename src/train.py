import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import FallingData
from model.model import Encoder

# a simple train loop for development

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = FallingData("/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data")
dataloaders = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=1)
model = Encoder(num_classes=2).to(DEVICE)

for i_batch, sample_batched in enumerate(dataloaders):
      print(sample_batched.keys())