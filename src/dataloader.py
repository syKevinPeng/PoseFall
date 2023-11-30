import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
# import util functions from ./utils/utils.py
from utils.utils import euler_angles_to_matrix, rotation_6d_to_matrix

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
        label = self.label[self.label['Trial Number'] == trial_number]
        arm_rot = torch.tensor(data[["arm_loc_x","arm_loc_y","arm_loc_z"]].values)
        print(f'length of this action: {len(data)}')
        # euler angles to rotation matrix
        arm_rot = euler_angles_to_matrix(arm_rot, "XYZ")
        # rotation matrix to 6D representation
        arm_rot = rotation_6d_to_matrix(arm_rot)
        print(f'arm_rot: {arm_rot.shape}')


        data_dict = {
            "armature_rotation": torch.tensor(data[["arm_rot_x","arm_rot_y","arm_rot_z"]].values),
            "armature_location": torch.tensor(data[["arm_loc_x","arm_loc_y","arm_loc_z"]].values),
            "joint_rotation": torch.tensor(data.loc[:, "Pelvis_x":"R_Hand_z"].values),
            "label": torch.tensor((label.values)[0, 1:]) # remove the first column, which is the trial number
        }
        
        return data_dict
    
# test dataset works
data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
dataset = FallingData(data_path)
# print(f'length of dataset: {len(dataset)}')
print(f'first item in dataset: {dataset[0]}')
