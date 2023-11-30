import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class FallingData(Dataset):
    def __init__(self, data_path):
        if not Path(data_path).exists():
           raise FileNotFoundError(f'{data_path} does not exist')
        # find all csv files in the directory using pathlib
        self.data_path = sorted([f for f in Path(data_path).glob('Trial_*.csv')])
        self.label_path = Path(data_path) / 'label.csv'
        if not self.label_path.exists():
            raise FileNotFoundError(f'{self.label_path} does not exist')
        self.label = pd.read_csv(self.label_path)

        
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        path = self.data_path[idx]
        trial_number = int(path.stem.split('_')[1])
        data = pd.read_csv(path)
        data_dict = {
            "armature_rotation": torch.tensor(data[["arm_rot_x","arm_rot_y","arm_rot_z"]].values),
            "armature_location": torch.tensor(data[["arm_loc_x","arm_loc_y","arm_loc_z"]].values),
            "joint_rotation": torch.tensor(data.loc[:, "Pelvis_x":"R_Hand_z"].values),
        }
        
        return data
