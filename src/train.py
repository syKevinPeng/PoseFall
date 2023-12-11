import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import FallingData
from model.model import Encoder, Decoder
from model.CVAE import CAVE
from icecream import ic

# a simple train loop for development
# torch.autograd.set_detect_anomaly(True)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i_batch, (data_dict) in enumerate(dataloaders):
      optimizer.zero_grad()
      # # labels
      # impa_label = data_dict['impa_label'].to(DEVICE)
      # glit_label = data_dict['glit_label'].to(DEVICE)
      # fall_label = data_dict['fall_label'].to(DEVICE)
      # # poses
      # impact_poses = data_dict['impa_combined_poses'].to(DEVICE)
      # glit_poses = data_dict['glit_combined_poses'].to(DEVICE)
      # fall_poses = data_dict['fall_combined_poses'].to(DEVICE) 
      # # mask
      # impact_mask = data_dict['impa_src_key_padding_mask'].to(DEVICE)
      # glit_mask = data_dict['glit_src_key_padding_mask'].to(DEVICE)
      # fall_mask = data_dict['fall_src_key_padding_mask'].to(DEVICE)
      # # length
      # impact_length = data_dict['impa_lengths'].to(DEVICE)
      # glit_length = data_dict['glit_lengths'].to(DEVICE)
      # fall_length = data_dict['fall_lengths'].to(DEVICE)

      # for label, pose, mask in zip([impa_label, glit_label, fall_label], [impact_poses, glit_poses, fall_poses], [impact_mask, glit_mask, fall_mask]):
      #       # construct the transformer
      #       num_class = label.size(1)
      #       ic(num_class)
            
      # exit()

      # build impact only
      # construct the transformer

      # batch = {
      #       'data': impact_poses, 
      #       'label': impa_label, 
      #       'mask': impact_mask, 
      #       "lengths": impact_length}
      batch = model(batch = data_dict)
      loss = model.compute_loss(batch)
      loss.backward()
      optimizer.step()
      ic(loss.item())
