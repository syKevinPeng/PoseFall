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

for i_batch, (data_dict, data_list) in enumerate(dataloaders):
      # labels
      impa_label = data_dict['impa_label'].to(DEVICE)
      glit_label = data_dict['glit_label'].to(DEVICE)
      fall_label = data_dict['fall_label'].to(DEVICE)
      # poses
      impact_poses = data_list[0].to(DEVICE)
      glit_poses = data_list[1].to(DEVICE)
      fall_poses = data_list[2].to(DEVICE) 
      # mask
      print(data_dict.keys())
      impact_mask = data_dict['impa_src_key_padding_mask'].to(DEVICE)
      glit_mask = data_dict['glit_src_key_padding_mask'].to(DEVICE)
      fall_mask = data_dict['fall_src_key_padding_mask'].to(DEVICE)

      print(impact_poses.shape)