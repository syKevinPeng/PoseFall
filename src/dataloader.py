from random import randint
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from .data_processing.utils import euler_angles_to_matrix, matrix_to_rotation_6d
import numpy as np
from icecream import ic

impact_phase_att = [
    "Impact Location",
    "Impact Attribute",
]  # full list: ["Impact Location", "Impact Attribute", "Impact Force"]
glitch_phase_att = [
    "Glitch Attribute"
]  # full list: ["Glitch Speed", "Glitch Attribute"]
fall_phase_att = ["Fall Attribute"]  # full list ["Fall Attribute","End Postion"]


class FallingDataset3Phase(Dataset):
    """
    For three phase data loading
    """

    def __init__(
        self,
        data_path,
        max_frame_dict,
        data_aug=True,
        sampling_every_n_frames=2,
        phase=["impa", "glit", "fall"],
        split="all",
    ):
        if not Path(data_path).exists():
            raise FileNotFoundError(f"{data_path} does not exist")
        # find all csv files in the directory using pathlib
        self.data_path = sorted([f for f in Path(data_path).glob("Trial_*.csv")])
        if split == "train":
            self.data_path = self.data_path[: int(len(self.data_path) * 0.7)]
        elif split == "eval":
            self.data_path = self.data_path[int(len(self.data_path) * 0.7) :]
        elif split == "all":
            pass
        else:
            raise ValueError(f"split {split} is not supported")
        print(f"Loading {split} split")
        self.label_path = Path(data_path) / "label.csv"
        if not self.label_path.exists():
            raise FileNotFoundError(f"{self.label_path} does not exist")
        # processing the label and only select needed col/attributes
        self.label = pd.read_csv(self.label_path)
        label_col = self.label.columns.to_list()
        impact_label = ["Trial Number"] + [
            col
            for col in label_col
            if any(col.startswith(att) for att in impact_phase_att)
        ]
        glitch_label = ["Trial Number"] + [
            col
            for col in label_col
            if any(col.startswith(att) for att in glitch_phase_att)
        ]
        fall_label = ["Trial Number"] + [
            col
            for col in label_col
            if any(col.startswith(att) for att in fall_phase_att)
        ]
        self.impact_label = self.label[impact_label]
        self.glitch_label = self.label[glitch_label]
        self.fall_label = self.label[fall_label]

        self.phase = phase
        self.max_fram_dict = max_frame_dict
        self.data_aug = data_aug
        self.sampling_every_n_frames = sampling_every_n_frames

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        path = self.data_path[idx]
        trial_number = int(path.stem.split("_")[1])
        data = pd.read_csv(path, header=0)
        # randomly select a start frame
        start_frame = randint(0, self.sampling_every_n_frames)
        data = data[start_frame :: self.sampling_every_n_frames]

        data_dict = {
            "impa_label": torch.tensor(
                self.impact_label[self.impact_label["Trial Number"] == trial_number]
                .iloc[:, 1:]
                .values,
                dtype=torch.float,
            ).flatten(),
            "glit_label": torch.tensor(
                self.glitch_label[self.glitch_label["Trial Number"] == trial_number]
                .iloc[:, 1:]
                .values,
                dtype=torch.float,
            ).flatten(),
            "fall_label": torch.tensor(
                self.fall_label[self.fall_label["Trial Number"] == trial_number]
                .iloc[:, 1:]
                .values,
                dtype=torch.float,
            ).flatten(),
        }
        # set the starting location of the armature to be the origin
        data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]] = (
            data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]]
            - data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].iloc[0]
        )
        # selec the data that contains action phase information
        for phase in self.phase:
            phase_data = data[data["phase"] == phase]
            # frame length
            frame_length = len(phase_data)
            # Data Augmentation: apply fft transformation
            if self.data_aug:
                # apply fft transformation
                fft_data = np.fft.fft(
                    phase_data.iloc[:, 1:-1].values.astype(float), axis=0
                )
                # randomly augment the data by the scale factor of 0.9-1.1
                scale_factor = np.random.uniform(0.9, 1.1)
                magnitudes = np.abs(fft_data) * scale_factor
                phases = np.angle(fft_data)
                scaled_fft_data = magnitudes * np.exp(1j * phases)
                # inverse fft
                sacled_bone_rot = np.fft.ifft(scaled_fft_data, axis=0)
                augmented_data = sacled_bone_rot.real.astype(float)
                phase_data = pd.DataFrame(
                    data=augmented_data, columns=phase_data.columns[1:-1]
                )

            if frame_length == 0:
                ic(phase_data.head())
                raise ValueError(f"{phase} frame length is 0. Data path is {path}")
            # process armature rotation
            arm_rot = torch.tensor(
                phase_data[["arm_rot_x", "arm_rot_y", "arm_rot_z"]].values
            )
            # euler angles to rotation matrix
            arm_rot = euler_angles_to_matrix(arm_rot, "XYZ")
            # rotation matrix to 6D representation
            arm_rot = matrix_to_rotation_6d(arm_rot)

            # armature location
            arm_loc = torch.tensor(
                phase_data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].values
            )
            padded_arm_loc = torch.zeros((len(arm_loc), 6))
            padded_arm_loc[:, :3] = arm_loc
            # extend one dim
            padded_arm_loc = padded_arm_loc.unsqueeze(1)
            # process bone rotation
            bone_rot = torch.tensor(
                phase_data.loc[:, "Pelvis_x":"R_Hand_z"].values
            )  # shape(num_frames, 72)
            bone_rot = bone_rot.reshape(-1, 24, 3)
            bone_rot = euler_angles_to_matrix(bone_rot, "XYZ")
            bone_rot = matrix_to_rotation_6d(bone_rot)
            combined_pose = torch.cat(
                (
                    bone_rot,
                    padded_arm_loc,
                    arm_rot.unsqueeze(1),
                ),
                dim=1,
            ).float()
            curr_frame_length, num_of_joints, feat_dim = combined_pose.size()
            max_frame = self.max_fram_dict[phase]
            pad_length = max_frame - curr_frame_length
            if pad_length < 0:
                raise ValueError(
                    f"frame length {curr_frame_length} is greater than max frame length {max_frame}"
                )
            # pad the data to max frame length if the frame length is less than max frame length
            padded_combined_pose = torch.cat(
                (combined_pose, torch.zeros((pad_length, num_of_joints, feat_dim)))
            )
            data_dict[f"{phase}_combined_poses"] = padded_combined_pose
            src_key_padding_mask = torch.concat(
                [torch.zeros(curr_frame_length), torch.ones(pad_length)]
            )

            # sanity check if the padding mask are not all ones
            if torch.all(src_key_padding_mask == 1):
                ic(frame_length)
                # ic(pad_length)
                ic(src_key_padding_mask)
                raise ValueError(
                    f"{phase} src_key_padding_mask is all ones, please check"
                )
            if torch.all(src_key_padding_mask == 1):
                raise ValueError(f"src_key_padding_mask is all ones, please check")
            data_dict[f"{phase}_src_key_padding_mask"] = src_key_padding_mask

        return data_dict


class FallingDataset1Phase(FallingDataset3Phase):
    "For single phase data loading"

    def __init__(
        self,
        data_path,
        max_frame_dict,
        data_aug=True,
        sampling_every_n_frames=2,
        padding=True,
        split="all",
    ):
        super().__init__(data_path, data_aug, sampling_every_n_frames, split=split)
        self.phase = "combined"
        self.data_aug = data_aug
        self.sampling_every_n_frames = sampling_every_n_frames
        self.max_frame_dict = max_frame_dict
        self.padding = padding
        # self.onehot_label = onehot_label

        self.label_col = np.concatenate(
            [impact_phase_att, glitch_phase_att, fall_phase_att]
        )
        # print(f'label_col: {label_col}')

        col_names = [
            att
            for att in self.label.columns[1:]
            for col in self.label_col
            if att.startswith(col)
        ]

        self.label_class_length = [cls.split("_")[0] for cls in col_names]
        from collections import Counter

        self.label_class_length = [
            count for count in Counter(self.label_class_length).values()
        ]

        self.onehot_label_df = self.label[["Trial Number"] + col_names]

    def get_attr_size(self):
        return self.label_class_length

    def __getitem__(self, idx):
        path = self.data_path[idx]
        trial_number = int(path.stem.split("_")[1])
        data = pd.read_csv(path, header=0)
        data = data[data["phase"] != "none"]
        start_frame = randint(0, self.sampling_every_n_frames)
        data = data[start_frame :: self.sampling_every_n_frames]
        # apply fft transformation
        fft_data = np.fft.fft(data.iloc[:, 1:-1].values.astype(float), axis=0)
        # randomly augment the data by the scale factor of 0.9-1.1
        scale_factor = np.random.uniform(0.9, 1.1)
        magnitudes = np.abs(fft_data) * scale_factor
        phases = np.angle(fft_data)
        scaled_fft_data = magnitudes * np.exp(1j * phases)
        # inverse fft
        sacled_bone_rot = np.fft.ifft(scaled_fft_data, axis=0)
        augmented_data = sacled_bone_rot.real.astype(float)
        data = pd.DataFrame(data=augmented_data, columns=data.columns[1:-1])

        # if self.onehot_label:
        label = (
            self.onehot_label_df[self.onehot_label_df["Trial Number"] == trial_number]
            .iloc[:, 1:]
            .values.squeeze()
        )
        # else:
        #     label = self.category_label_df[self.category_label_df["Trial Number"] == trial_number].iloc[:, 1:].values.squeeze()
        data_dict = {
            f"{self.phase}_label": torch.tensor(label, dtype=torch.float),
        }

        # set the starting location of the armature to be the origin
        data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]] = (
            data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]]
            - data[["arm_loc_x", "arm_loc_y", "arm_loc_z"]].iloc[0]
        )

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
        # process bone rotation
        bone_rot = torch.tensor(
            data.loc[:, "Pelvis_x":"R_Hand_z"].values
        )  # shape(num_frames, 72)
        bone_rot = bone_rot.reshape(-1, 24, 3)
        bone_rot = euler_angles_to_matrix(bone_rot, "XYZ")
        bone_rot = matrix_to_rotation_6d(bone_rot)
        combined_pose = torch.cat(
            (
                bone_rot,
                padded_arm_loc,
                arm_rot.unsqueeze(1),
            ),
            dim=1,
        ).float()
        curr_frame_length, num_of_joints, feat_dim = combined_pose.size()
        if self.padding:
            # processing padding
            max_frame = sum([value for value in self.max_frame_dict.values()])
            pad_length = max_frame - curr_frame_length
            if pad_length < 0:
                raise ValueError(
                    f"frame length {curr_frame_length} is greater than max frame length {max_frame}"
                )
            # pad the data to max frame length if the frame length is less than max frame length
            padded_combined_pose = torch.cat(
                (combined_pose, torch.zeros((pad_length, num_of_joints, feat_dim)))
            )
            data_dict[f"{self.phase}_combined_poses"] = padded_combined_pose
            # prepare padding mask for the data: attend zero position and ignore none-zeros
            src_key_padding_mask = torch.concat(
                [torch.zeros(curr_frame_length), torch.ones(pad_length)]
            )
            # sanity check if the padding mask are not all ones
        else:
            data_dict[f"{self.phase}_combined_poses"] = combined_pose
            src_key_padding_mask = torch.zeros(curr_frame_length)
        if torch.all(src_key_padding_mask == 1):
            raise ValueError(f"src_key_padding_mask is all ones, please check")
        data_dict[f"{self.phase}_src_key_padding_mask"] = src_key_padding_mask
        return data_dict

    def collate_fn(self, batch):
        """
        Collate function for the dataset
        """
        combined_poses = torch.stack(
            [item[f"{self.phase}_combined_poses"] for item in batch]
        )
        src_key_padding_mask = torch.stack(
            [item[f"{self.phase}_src_key_padding_mask"] for item in batch]
        )
        labels = torch.stack([item[f"{self.phase}_label"] for item in batch])
        print(f"combined_poses: {combined_poses.size()}")
        return {
            f"{self.phase}_combined_poses": combined_poses,
            f"{self.phase}_src_key_padding_mask": src_key_padding_mask,
            f"{self.phase}_label": labels,
        }


# # test dataset works
# data_path = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data"
# dataset = myFallingDataset(data_path)
# # print(f'length of dataset: {len(dataset)}')
# print(dataset[0][f"combined_poses"].size())
