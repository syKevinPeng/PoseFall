import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import FallingData
from model.model import Encoder, Decoder
from model.CVAE import CAVE
from icecream import ic

# a simple train loop for development
torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHASES = ["impa", "glit", "fall"]
data = FallingData("/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data")
dataloaders = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

# get number of clases for each of the phase
impa_label = data[0]['impa_label']
glit_label = data[0]['glit_label']
fall_label = data[0]['fall_label']
num_class = {
      'impa': impa_label.size(0),
      'glit': glit_label.size(0),
      'fall': fall_label.size(0)
} 
model = CAVE(phase_names=PHASES, num_classes_dict=num_class).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for i_batch, (data_dict) in enumerate(dataloaders):
      optimizer.zero_grad()
      batch = model(batch = data_dict)
      loss = model.compute_all_phase_loss(batch)
      loss.backward()
      optimizer.step()
      ic(loss.item())
