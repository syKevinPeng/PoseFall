import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from data_processing.utils import euler_angles_to_matrix, matrix_to_rotation_6d
import numpy as np

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

        self.phase = ["impa", "glit", "fall"]
        # set the max frame length for each phase
        self.max_frame = {"impa": 120, "glit": 300, "fall": 100}

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        path = self.data_path[idx]
        trial_number = int(path.stem.split('_')[1])
        data = pd.read_csv(path)
        label = self.label[self.label['Trial Number'] == trial_number]
        data_dict = {"label": torch.tensor((label.values)[0, 1:])} # remove the first column, which is the trial number; shape(32)}
        # selec the data that contains action phase information
        phase = data[["phase"]].to_numpy()
        for phase in self.phase:
            data = data[data["phase"] == phase]
            # frame length
            frame_length = len(data)
            # process armature rotation
            arm_rot = torch.tensor(data[["arm_loc_x","arm_loc_y","arm_loc_z"]].values)
            # euler angles to rotation matrix
            arm_rot = euler_angles_to_matrix(arm_rot, "XYZ")
            # rotation matrix to 6D representation
            arm_rot = matrix_to_rotation_6d(arm_rot)

            # armature location
            arm_loc = torch.tensor(data[["arm_loc_x","arm_loc_y","arm_loc_z"]].values)
            # process bone rotation
            bone_rot = torch.tensor(data.loc[:, "Pelvis_x":"R_Hand_z"].values) # shape(num_frames, 72)
            bone_rot = bone_rot.reshape(-1, 24, 3)
            bone_rot = euler_angles_to_matrix(bone_rot, "XYZ")
            bone_rot = matrix_to_rotation_6d(bone_rot)
            
            # pad the data to max frame length if the frame length is less than max frame length
            pad_length = self.max_frame[phase] - frame_length
            if pad_length < 0:
                raise ValueError(f'{phase} frame length {frame_length} is greater than max frame length {self.max_frame[phase]}')
            arm_rot = torch.cat((arm_rot, torch.zeros((pad_length, 6))))
            arm_loc = torch.cat((arm_loc, torch.zeros((pad_length, 3))))
            bone_rot = torch.cat((bone_rot, torch.zeros((pad_length, 24, 6))))
            
            data_dict[f'{phase}_armature_rotation'] = arm_rot # shape(num_frames, 6)
            data_dict[f'{phase}_armature_location'] = arm_loc # shape(num_frames, 3)
            data_dict[f'{phase}_joint_rotation'] = bone_rot # shape(num_frames, 24,6)
        return data_dict
    
# test dataset works
# data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
# dataset = FallingData(data_path)
# # print(f'length of dataset: {len(dataset)}')
# print(dataset[0]['armature_rotation'].size())
# print(dataset[0]['armature_location'].size())
# print(dataset[0]['joint_rotation'].size())
# print(dataset[0]['label'].size())
