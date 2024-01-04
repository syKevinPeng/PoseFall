import enum
from sympy import false
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from data_processing.utils import euler_angles_to_matrix, matrix_to_rotation_6d
import numpy as np
from icecream import ic

impact_phase_att = ["Impact Location", "Impact Attribute"] # full list: ["Impact Location", "Impact Attribute", "Impact Force"]
glitch_phase_att = ["Glitch Attribute"] # full list: ["Glitch Speed", "Glitch Attribute"]
fall_phase_att = ["Fall Attribute","End Postion"]

class FallingData(Dataset):
    def __init__(self, data_path, data_aug = True, sampling_rate = 60):
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
        self.max_frame = {"impa": 120, "glit": 350, "fall": 400}
        self.data_aug = data_aug


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        path = self.data_path[idx]
        trial_number = int(path.stem.split("_")[1])
        data = pd.read_csv(path, header=0)
        # sample 60 frames from all the index. Add a small random number to avoid sampling the same frame
        index = np.linspace(10, len(data)-10, num=60, dtype=int)
        random_shift = np.random.randint(-5, 5, size=60)
        index = index + random_shift
        # sort the index just in case
        index.sort()
        data = data.iloc[index]

        # col_name =  data.columns[1:-1].to_numpy()
        # col_name = col_name.reshape(-1, 3)[2:, :]
        # print(col_name.shape)


        # exit()
        data_dict ={
            "frame_num": torch.tensor(max(data["frame"].values)),
            "impa_label": torch.tensor(self.impact_label[self.impact_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
            "glit_label": torch.tensor(self.glitch_label[self.glitch_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
            "fall_label": torch.tensor(self.fall_label[self.fall_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
        }
        # set the starting location of the armature to be the origin
        data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]] = data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]] - data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].iloc[0]
        # selec the data that contains action phase information
        for phase in self.phase:
            phase_data = data[data["phase"] == phase]
            # frame length
            frame_length = len(phase_data)
            # Data Augmentation: apply fft transformation
            if self.data_aug:
                # apply fft transformation
                fft_data = np.fft.fft(phase_data.iloc[:, 1:-1].values.astype(float), axis=0)
                # randomly augment the data by the scale factor of 0.9-1.1
                scale_factor = np.random.uniform(0.9, 1.1)
                magnitudes = np.abs(fft_data)*scale_factor
                phases = np.angle(fft_data)
                scaled_fft_data = magnitudes * np.exp(1j*phases)
                # inverse fft
                sacled_bone_rot = np.fft.ifft(scaled_fft_data, axis=0)
                augmented_data = sacled_bone_rot.real.astype(float)
                phase_data = pd.DataFrame(data=augmented_data, columns=phase_data.columns[1:-1])
                
            if frame_length == 0:
                ic(phase_data.head())
                raise ValueError(f"{phase} frame length is 0. Data path is {path}")
            # process armature rotation
            arm_rot = torch.tensor(phase_data[["arm_rot_x", "arm_rot_y", "arm_rot_z"]].values)
            # euler angles to rotation matrix
            arm_rot = euler_angles_to_matrix(arm_rot, "XYZ")
            # rotation matrix to 6D representation
            arm_rot = matrix_to_rotation_6d(arm_rot)

            # armature location
            arm_loc = torch.tensor(phase_data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].values)
            # set the starting location of the armature to be the origin
            # process bone rotation
            bone_rot = torch.tensor(
                phase_data.loc[:, "Pelvis_x":"R_Hand_z"].values
            )  # shape(num_frames, 72)
            bone_rot = bone_rot.reshape(-1, 24, 3)
            bone_rot = euler_angles_to_matrix(bone_rot, "XYZ")
            bone_rot = matrix_to_rotation_6d(bone_rot)

            # # pad the data to max frame length if the frame length is less than max frame length
            # pad_length = self.max_frame[phase] - frame_length
            # if pad_length < 0:
            #     raise ValueError(
            #         f"{phase} frame length {frame_length} is greater than max frame length {self.max_frame[phase]}"
            #     )
            # arm_rot = torch.cat((arm_rot, torch.zeros((pad_length, 6))))
            # arm_loc = torch.cat((arm_loc, torch.zeros((pad_length, 3))))
            # bone_rot = torch.cat((bone_rot, torch.zeros((pad_length, 24, 6))))

            # # prepare padding mask for the data: attend zero position and ignore none-zeros
            # # TODO: simplify the padding mask
            # src_key_padding_mask = torch.cat(
            #     (torch.zeros((frame_length)), torch.ones((pad_length)))
            # )
            src_key_padding_mask = torch.zeros((frame_length))

            # sanity check if the padding mask are not all ones
            if torch.all(src_key_padding_mask == 1):
                ic(frame_length)
                # ic(pad_length)
                ic(src_key_padding_mask)
                raise ValueError(
                    f"{phase} src_key_padding_mask is all ones, please check"
                )
            data_dict[f"{phase}_armature_rotation"] = arm_rot  # shape(num_frames, 6)
            data_dict[f"{phase}_armature_location"] = arm_loc  # shape(num_frames, 3)
            data_dict[f"{phase}_joint_rotation"] = bone_rot.reshape(
                self.max_frame[phase], -1
            )  # shape(num_frames, 24,6) => shape(num_frames, 144)

            data_dict[f"{phase}_src_key_padding_mask"] = src_key_padding_mask # shape(num_frames,)
            data_dict[f"{phase}_lengths"] = torch.tensor(frame_length)  # Note: this the length of the actual sequence, not the padded one.
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
    


class myFallingData(FallingData):
    def __init__(self, data_path, data_aug = True, sampling_rate = 60):
        super().__init__(data_path, data_aug, sampling_rate)
        self.phase = ["impa", "glit", "fall"]
        self.data_aug = data_aug
        self.sampling_rate = sampling_rate

    def __getitem__(self, idx):
        path = self.data_path[idx]
        trial_number = int(path.stem.split("_")[1])
        data = pd.read_csv(path, header=0)
        # sample 60 frames from all the index. Add a small random number to avoid sampling the same frame
        index = np.linspace(10, len(data)-10, num=self.sampling_rate, dtype=int)
        random_shift = np.random.randint(-5, 5, size=self.sampling_rate)
        index = index + random_shift
        # sort the index just in case
        index.sort()
        data = data.iloc[index]

        data_dict ={
            "frame_num": torch.tensor(max(data["frame"].values)),
            "impa_label": torch.tensor(self.impact_label[self.impact_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
            "glit_label": torch.tensor(self.glitch_label[self.glitch_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
            "fall_label": torch.tensor(self.fall_label[self.fall_label["Trial Number"] == trial_number].iloc[:,1:].values, dtype=torch.float).flatten(),
        }
        # set the starting location of the armature to be the origin
        data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]] = data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]] - data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].iloc[0]
        # selec the data that contains action phase information
        # process armature rotation
        arm_rot = torch.tensor(data[["arm_rot_x", "arm_rot_y", "arm_rot_z"]].values)
        # euler angles to rotation matrix
        arm_rot = euler_angles_to_matrix(arm_rot, "XYZ")
        # rotation matrix to 6D representation
        arm_rot = matrix_to_rotation_6d(arm_rot)

        # armature location
        arm_loc = torch.tensor(data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].values)
        padded_arm_loc = torch.zeros((len(arm_loc), 6))
        padded_arm_loc[:, :3] = arm_loc
        # extend one dim
        padded_arm_loc = padded_arm_loc.unsqueeze(1)
        # set the starting location of the armature to be the origin
        # process bone rotation
        bone_rot = torch.tensor(
            data.loc[:, "Pelvis_x":"R_Hand_z"].values
        )  # shape(num_frames, 72)
        bone_rot = bone_rot.reshape(-1, 24, 3)
        bone_rot = euler_angles_to_matrix(bone_rot, "XYZ")
        bone_rot = matrix_to_rotation_6d(bone_rot) 
        src_key_padding_mask = torch.zeros(len(data))

        # sanity check if the padding mask are not all ones
        if torch.all(src_key_padding_mask == 1):
            # ic(frame_length)
            # ic(pad_length)
            ic(src_key_padding_mask)
            raise ValueError(
                f"src_key_padding_mask is all ones, please check"
            )
        # data_dict[f"armature_rotation"] = arm_rot  # shape(num_frames, 6)
        # data_dict[f"armature_location"] = arm_loc  # shape(num_frames, 3)
        # data_dict[f"joint_rotation"] = bone_rot
        data_dict[f"src_key_padding_mask"] = src_key_padding_mask # shape(num_frames,)# Note: this the length of the actual sequence, not the padded one.
        combined_pose = torch.cat(
            (
                bone_rot,
                padded_arm_loc,
                arm_rot.unsqueeze(1),
            ),
            dim=1,
        ).float()
        data_dict[f"combined_poses"] = combined_pose
        return data_dict    
        


# test dataset works
data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
dataset = myFallingData(data_path)
# print(f'length of dataset: {len(dataset)}')
print(dataset[0][f"combined_poses"].size())
