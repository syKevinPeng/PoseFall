import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from data_processing.utils import euler_angles_to_matrix, matrix_to_rotation_6d

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
        # process armature rotation
        arm_rot = torch.tensor(data[["arm_loc_x","arm_loc_y","arm_loc_z"]].values)
        # euler angles to rotation matrix
        arm_rot = euler_angles_to_matrix(arm_rot, "XYZ")
        # rotation matrix to 6D representation
        arm_rot = matrix_to_rotation_6d(arm_rot)
        # process bone rotation
        bone_rot = torch.tensor(data.loc[:, "Pelvis_x":"R_Hand_z"].values) # shape(num_frames, 72)
        bone_rot = bone_rot.reshape(-1, 24, 3)
        bone_rot = euler_angles_to_matrix(bone_rot, "XYZ")
        bone_rot = matrix_to_rotation_6d(bone_rot)
        
        data_dict = {
            "armature_rotation": arm_rot, # shape(num_frames, 6)
            "armature_location": torch.tensor(data[["arm_loc_x","arm_loc_y","arm_loc_z"]].values), # shape(num_frames, 3)
            "joint_rotation": bone_rot, # shape(num_frames, 24,6)
            "label": torch.tensor((label.values)[0, 1:]) # remove the first column, which is the trial number; shape(32)
        }
        
        return data_dict
    
# test dataset works
data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
dataset = FallingData(data_path)
# print(f'length of dataset: {len(dataset)}')
print(dataset[0]['armature_rotation'].size())
print(dataset[0]['armature_location'].size())
print(dataset[0]['joint_rotation'].size())
print(dataset[0]['label'].size())
