# %%
import numpy as np
from pathlib import Path
from icecream import ic
from scipy.datasets import face
# %%
DATAPATH = '/home/siyuan/research/PoseFall/data/falling_dataset.npz'
VIZ_OUTPUT = '/home/siyuan/research/PoseFall/src/visulization/viz_output'
DATAPATH = Path(DATAPATH)
VIZ_OUTPUT = Path(VIZ_OUTPUT)
if DATAPATH.is_file():
    dataset = np.load(DATAPATH, allow_pickle=True)['falling_dataset'][()]
else:
    raise FileNotFoundError('Data file not found.')

if not VIZ_OUTPUT.is_dir():
    VIZ_OUTPUT.mkdir()
# %%
# second trial:
point_trace = dataset[2] # frame, num_joints, 3
num_frame, num_joints, _ = point_trace.shape
SMPL_num_joints = 23
# add two new point to trace
point_trace = np.concatenate([point_trace, np.zeros((num_frame, 3, 3))], axis=1) # jaw, left hand, right hand]

# recorder the joint index
joint_index = np.array([14, 7, 2, 5, 18, 19, 3,1, 20, 16, 12, 21, 13, 8, 11, 23, 10, 22, 15, 4,9, 6, 24, 25, 17 ])
joint_index = joint_index - 1

# record index
point_trace = point_trace[:, joint_index, :]




# %%
'''
SMPL Config
0: 'pelvis',
1: 'left_hip',
2: 'right_hip',
3: 'spine1',
4: 'left_knee',
5: 'right_knee',
6: 'spine2',
7: 'left_ankle',
8: 'right_ankle',
9: 'spine3',
10: 'left_foot',
11: 'right_foot',
12: 'neck',
13: 'left_collar',
14: 'right_collar',
15: 'head',
16: 'left_shoulder',
17: 'right_shoulder',
18: 'left_elbow',
19: 'right_elbow',
20: 'left_wrist',
21: 'right_wrist',
22: 'left_hand',
23: 'right_hand'
'''
# %%
import smplx
# from smplx.joint_names import  JOINT_NAMES, SMPL_JOINT_NAMES 
model_folder= '/home/siyuan/research/PoseFall/data/SMPL_cleaned'
human_model = smplx.SMPL(
    model_path = model_folder,
    create_body_pose=True,
    body_pose= None,
    create_betas=True,
    betas=None,
    gender='male'
)
# get mesh and joints from the SMPL model
smplx_output = human_model()
joints = smplx_output.joints
joints = joints[:24] # remove the extra joints "The remaining 21 points are vertices selected to match some 2D keypoint annotations

# %%
from smplx.joint_names import SMPLH_JOINT_NAMES
from smplx.joint_names import SMPL_JOINT_NAMES 

# select SMPL joints from SMPLH joint
# SMPL_joint_index = [SMPLH_JOINT_NAMES.index(name) if name in for name in SMPL_JOINT_NAMES]
# SMPL_joint_index = np.array(SMPL_joint_index)
# SMPL_joint_index
# %%
