from smplx import SMPLLayer
import torch.nn as nn
from data_processing.utils import parse_output
from icecream import ic


class SMPLModel(nn.Module):
    """
    Using SMPL layer to generate vertex locations for the human body
    """
    def __init__(self, SMPL_model_path = "/home/siyuan/research/PoseFall/data/SMPL_cleaned/SMPL_FEMALE.pkl") -> None:
        super().__init__()
        # initialize SMPL model
        self.human_model = SMPLLayer(
            model_path = SMPL_model_path,
            creat_body_pose=True,
        )

    def forward(self, batch):
        """
        we discard the locationa and rotation of the armature. 
        Only calculate the vertex transformation resulted from joint rotation
        """

        parsed_output = parse_output(batch)
        bone_rot = parsed_output["bone_rot"]
        ic(bone_rot.shape)
        # convert continuous rotation representationt to rotation matrix
        

        exit()
