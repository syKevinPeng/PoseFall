# %%
import numpy as np
from pathlib import Path
from icecream import ic
from scipy.datasets import face
from icecream import ic
import torch
# %%
DATAPATH = '/home/siyuan/research/PoseFall/data/falling_dataset_Cam1.npz'
VIZ_OUTPUT = '/home/siyuan/research/PoseFall/src/visulization/viz_output'
DATAPATH = Path(DATAPATH)
VIZ_OUTPUT = Path(VIZ_OUTPUT)
if DATAPATH.is_file():
    dataset_pos = np.load(DATAPATH, allow_pickle=True)['falling_dataset_pos'][()]
    dataset_rot = np.load(DATAPATH, allow_pickle=True)['falling_dataset_rot'][()]
else:
    raise FileNotFoundError('Data file not found.')

if not VIZ_OUTPUT.is_dir():
    VIZ_OUTPUT.mkdir()
# %%
dataset_pos.keys()
# %%
# Trial 3
joint_locs = dataset_pos["Trial3"] # frame, num_joints, 3
joint_rots = dataset_rot["Trial3"] # frame, num_joints, 3, 3
# num_frame, num_joints, _ = joint_loc.shape

# # get one frame
# frame = joint_loc[0, :, :]
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure( figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=0, azim=-90)
# ax.set_xlim(-0.5, 0.2)
# ax.set_ylim(0, 1)
# joints = frame
# ic(joints.shape)
# for i in range(joints.shape[0]):
#     ax.scatter(joints[i, 0], joints[i, 1], joints[i, 2], c='r', marker='o')
#     ax.text(joints[i, 0] + 0.02, joints[i, 1], joints[i, 2], str(i+1), fontsize = 10)


# # recorder the joint index
joint_index = np.array([16,3,9, 24, 18, 10, 5, 15, 4, 14, 13, 1, 19, 22, 20, 7, 17, 2, 21, 12, 8, 11, 6, 23])
# ic(joint_index.shape)
# ic(sorted(joint_index))

joint_index = joint_index - 1
#%%
# point trace shape: (frame, num_joints, 3)
ic(joint_locs.shape)
ic(joint_rots.shape)
# # record index
joint_locs = joint_locs[:, joint_index, :]
joint_rots = joint_rots[:, joint_index, :]

# get one frame
joint_loc = joint_locs[0, :, :]
joint_rot = joint_rots[0, :, :]

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
import pickle
# from smplx.joint_names import  JOINT_NAMES, SMPL_JOINT_NAMES 
model_folder= '/home/siyuan/research/PoseFall/data/SMPL_cleaned'
male_model = "/home/siyuan/research/PoseFall/data/SMPL_cleaned/SMPL_MALE.pkl"
# get kinematic tree
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

with open(male_model, 'rb') as smpl_file:
    data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))
parents = (data_struct.kintree_table[0]).astype(np.int32)
parents = torch.tensor(parents)
parents[0] = -1


human_model = smplx.SMPL(
    model_path = model_folder,
    create_body_pose=True,
    body_pose= None,
    create_betas=True,
    betas=None,
    gender='male',
)
output = human_model()
curr_joint_ori  = output.get('body_pose').detach().cpu().reshape(-1, 3)
new_joint_ori = joint_rot.reshape(1, -1)
print(f'new_joint_ori: {new_joint_ori.shape}')

# this_joint = joint_rot[0, :, :].flatten().reshape(1, -1)
smplx_output = human_model(body_pose=new_joint_ori)


joints = smplx_output.joints
joints = joints[:24] # remove the extra joints "The remaining 21 points are vertices selected to match some 2D keypoint annotations

# %%
from pose_viz import viz_360

output_file = "viz_output/pose1.png"
viz_360(output_file,human_model, smplx_output)