import torch
from torch.utils.data import Dataset
from pathlib import Path
from .stgcn import STGCN
import argparse, yaml
from tqdm import tqdm
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd
from ..data_processing.utils import euler_angles_to_matrix, matrix_to_rotation_6d
class EvaluateDataset(Dataset):
    def __init__(self, config, data_path, max_frame_dict, padding=True):
        self.config = config
        self.data_path = Path(data_path)
        self.max_frame_dict = max_frame_dict
        self.padding = padding

        if not self.data_path.is_dir():
            raise ValueError(f"{self.data_path} is not a directory.")
        
        self.data_list = list(self.data_path.glob("*.csv"))
        num_class = len(self.data_list[0].stem.split("_")[0])
        print(f"Number of classes: {num_class}")
        self.recognition_model =  STGCN(in_channels=6, 
                  num_class=num_class, 
                  graph_args={"layout": "smpl", "strategy": "spatial"},
                  edge_importance_weighting=True, 
                  device=DEVICE).to(DEVICE)

    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        label = [int(i) for i in path.stem.split("_")[0]]
        data_dict = {"label": torch.tensor(label)}
        # read csv file
        data = pd.read_csv(path)
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
            data_dict[f"combined_poses"] = padded_combined_pose
            # prepare padding mask for the data: attend zero position and ignore none-zeros
            src_key_padding_mask = torch.concat(
                [torch.zeros(curr_frame_length), torch.ones(pad_length)]
            )
            # sanity check if the padding mask are not all ones
        else:
            data_dict[f"combined_poses"] = combined_pose
            src_key_padding_mask = torch.zeros(curr_frame_length)
        if torch.all(src_key_padding_mask == 1):
            raise ValueError(f"src_key_padding_mask is all ones, please check")
        data_dict[f"src_key_padding_mask"] = src_key_padding_mask
        return data_dict

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default=Path(__file__).parent.parent.joinpath("config.yaml"),
        help="path to config file",
    )
    cmd_args = parser.parse_args()
    # load config file
    with open(cmd_args.config_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args

if __name__=="__main__":
    args = parse_args()
    # prepare wandb
    # wandb.init(
    #     project="posefall_recognition",
    #     config=args,
    #     mode=wandb_config["wandb_mode"],
    #     tags=wandb_config["wandb_tags"],
    #     name="recognition_training",
    #     notes="training recognition model with GT data",
    # )
    max_frame_dict = args["constant"]["max_frame_dict"]
    dataset = EvaluateDataset(args, args["evaluate_config"]["evaluate_dataset_path"], max_frame_dict=max_frame_dict)
    print(dataset[0])