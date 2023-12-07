import enum
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from data_processing.utils import euler_angles_to_matrix, matrix_to_rotation_6d
import numpy as np

impact_phase_att = ["Impact Location", "Impact Attribute"] # full list: ["Impact Location", "Impact Attribute", "Impact Force"]
glitch_phase_att = ["Glitch Attribute"] # full list: ["Glitch Speed", "Glitch Attribute"]
fall_phase_att = ["Fall Attribute","End Postion"]

class FallingData(Dataset):
    def __init__(self, data_path):
        if not Path(data_path).exists():
            raise FileNotFoundError(f"{data_path} does not exist")
        # find all csv files in the directory using pathlib
        self.data_path = sorted([f for f in Path(data_path).glob("Trial_*.csv")])
        self.label_path = Path(data_path) / "label.csv"
        if not self.label_path.exists():
            raise FileNotFoundError(f"{self.label_path} does not exist")
        
        # processing the label and only select needed col/attributes
        self.label = pd.read_csv(self.label_path)
        label_col = self.label.columns.to_list()
        impact_label = ["Trial Number"] + [col for col in label_col if any(col.startswith(att) for att in impact_phase_att)  ]
        glitch_label = ["Trial Number"] +[col for col in label_col if any(col.startswith(att) for att in glitch_phase_att)  ]
        fall_label = ["Trial Number"] + [col for col in label_col if any(col.startswith(att) for att in fall_phase_att)  ]
        self.impact_label = self.label[impact_label]
        self.glitch_label = self.label[glitch_label]
        self.fall_label = self.label[fall_label]

        self.phase = ["impa", "glit", "fall"]
        # set the max frame length for each phase
        self.max_frame = {"impa": 120, "glit": 300, "fall": 100}


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # TODO: reset the starting location of the armature to be the origin
        path = self.data_path[idx]
        trial_number = int(path.stem.split("_")[1])
        data = pd.read_csv(path)
        # label = self.label[self.label["Trial Number"] == trial_number]

        data_dict ={
            "frame": torch.tensor(data["frame"].values),
            "impa_label": torch.tensor(self.impact_label[self.impact_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
            "glit_label": torch.tensor(self.glitch_label[self.glitch_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
            "fall_label": torch.tensor(self.fall_label[self.fall_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
        }
        # selec the data that contains action phase information
        for phase in self.phase:
            data = data[data["phase"] == phase]
            # frame length
            frame_length = len(data)
            # process armature rotation
            arm_rot = torch.tensor(data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].values)
            # euler angles to rotation matrix
            arm_rot = euler_angles_to_matrix(arm_rot, "XYZ")
            # rotation matrix to 6D representation
            arm_rot = matrix_to_rotation_6d(arm_rot)

            # armature location
            arm_loc = torch.tensor(data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].values)
            # process bone rotation
            bone_rot = torch.tensor(
                data.loc[:, "Pelvis_x":"R_Hand_z"].values
            )  # shape(num_frames, 72)
            bone_rot = bone_rot.reshape(-1, 24, 3)
            bone_rot = euler_angles_to_matrix(bone_rot, "XYZ")
            bone_rot = matrix_to_rotation_6d(bone_rot)

            # pad the data to max frame length if the frame length is less than max frame length
            pad_length = self.max_frame[phase] - frame_length
            if pad_length < 0:
                raise ValueError(
                    f"{phase} frame length {frame_length} is greater than max frame length {self.max_frame[phase]}"
                )
            arm_rot = torch.cat((arm_rot, torch.zeros((pad_length, 6))))
            arm_loc = torch.cat((arm_loc, torch.zeros((pad_length, 3))))
            bone_rot = torch.cat((bone_rot, torch.zeros((pad_length, 24, 6))))

            # prepare padding mask for the data: attend zero position
            arm_rot_padding_mask = torch.cat(
                (torch.zeros((frame_length, 6)), torch.ones((pad_length, 6)))
            )
            arm_loc_padding_mask = torch.cat(
                (torch.zeros((frame_length, 3)), torch.ones((pad_length, 3)))
            )
            bone_rot_padding_mask = torch.cat(
                (torch.zeros((frame_length, 24, 6)), torch.ones((pad_length, 24, 6)))
            )

            data_dict[f"{phase}_armature_rotation"] = arm_rot  # shape(num_frames, 6)
            data_dict[f"{phase}_armature_location"] = arm_loc  # shape(num_frames, 3)
            data_dict[f"{phase}_joint_rotation"] = bone_rot.reshape(
                self.max_frame[phase], -1
            )  # shape(num_frames, 24,6) => shape(num_frames, 144)

            data_dict[f"{phase}_src_key_padding_mask"] = torch.cat(
                (
                    arm_rot_padding_mask,
                    arm_loc_padding_mask,
                    bone_rot_padding_mask.reshape(self.max_frame[phase], -1),
                ),
                dim=1,
                
            )  # shape(num_frames, 153)
            combined_pose = torch.cat(
                (
                    data_dict[f"{phase}_armature_location"],
                    data_dict[f"{phase}_armature_rotation"],
                    data_dict[f"{phase}_joint_rotation"],
                ),
                dim=1,
            ).float()
            data_dict[f"{phase}_combined_poses"] = combined_pose    


        return data_dict


# test dataset works
# data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
# dataset = FallingData(data_path)
# # print(f'length of dataset: {len(dataset)}')
# print(dataset[0]['armature_rotation'].size())
# print(dataset[0]['armature_location'].size())
# print(dataset[0]['joint_rotation'].size())
# print(dataset[0]['label'].size())
