# %%
import numpy as np
from pathlib import Path
from icecream import ic
from scipy.datasets import face
from icecream import ic
import torch
import visulization.blender_convert_to_smpl as blender_convert_to_smpl
import matplotlib.pyplot as plt
from smplx import joint_names
joint_names = joint_names.SMPL_JOINT_NAMES 

import smplx
import pickle
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

# Trial 3
trial_sequence = "Trial18"
joint_locs = dataset_pos[trial_sequence] # frame, num_joints, 3
joint_rots = dataset_rot[trial_sequence] # frame, num_joints, 3, 3

# # recorder the joint index
joint_index = np.array([16,3,9, 24, 18, 10, 5, 15, 4, 14, 13, 1, 19, 22, 20, 7, 17, 2, 21, 12, 8, 11, 6, 23])
joint_index = joint_index - 1
# # record index
joint_locs = joint_locs[:, joint_index, :]
joint_rots = joint_rots[:, joint_index, :]
# get one frame
joint_loc = joint_locs[0, :, :]
joint_rot = joint_rots[0, :, :]

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
parent_vec_list = []
for i in range(len(joint_loc)):
    # get the parent index
    parent_index = parents[i]
    # plot a line between the current joint and its parent
    if parent_index != -1:
        parent_vec_list.append(joint_loc[i, :] - joint_loc[parent_index, :])
    else:
        parent_vec_list.append(joint_loc[i, :] - [0,0,1])
# vector_list is from parent to child (i.e. the parent vector)
# print(parent_vec_list)

# child_vec_list = []
# # print(f'')
# for i in range(len(joint_loc)):
#     child_index = np.where(parents == i)
#     if len(child_index) == 0 or len(child_index) > 1:
#         child_vec_list.append([0,0,0])
#     else:
#         child_vec_list.append(joint_loc[child_index[0][0], :] - joint_loc[i, :])
#     print(f'child index is {child_index}')
# exit()

# given the direction vector of the joint (vector_list), visulize the joint orientation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=0, azim=-90, roll = 0)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 1)
# Plot each joint
for joint in joint_loc:
    ax.scatter(joint[0], joint[1], joint[2], color='blue', s=50)  # s is the size of the point
# Plot lines for each bone
for joint, vector in zip(joint_loc, parent_vec_list):
    line = np.vstack((joint, joint - vector))  # stack the start and end points vertically
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color='red')

for joint, rot in zip(joint_loc, joint_rot):
    # plot the rotation

    rot = blender_convert_to_smpl.euler_to_matrix(rot)
    rot = rot @ np.array([0,0,0.1])

    line = np.vstack((joint, joint + rot))  # stack the start and end points vertically
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color='green')


# Set plot labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Joint Locations and Bone Directions')
fig.savefig(VIZ_OUTPUT / 'joint_dir.png', dpi=200)

for name , rot in zip(joint_names, joint_rot):
    print(f'{name} rot is {rot}')
exit()




# # calculate the relative rotation
# relative_joint_ori = []
# for i in range(len(parent_vec_list)):
#     # get the parent index
#     parent_index = parents[i]
#     # plot a line between the current joint and its parent
#     if parent_index != -1:
#         relative_joint_ori.append(rot_convert.calculate_euler_angles(parent_vec_list[i], parent_vec_list[i]))
#     else:
#         relative_joint_ori.append(rot_convert.calculate_euler_angles([0,0,1], parent_vec_list[i]))
# # # convert global joint_ori to local joint_ori
# # relative_joint_ori = rot_convert.global_to_relative_euler(new_joint_ori, parents)

# relative_joint_ori = torch.tensor(relative_joint_ori, dtype=torch.float32)[1:, :] # remove the root joint
# # calibrate the axis

# test_joint_ori = relative_joint_ori
# # test_joint_ori = torch.zeros_like(relative_joint_ori)
# test_joint_ori[15, :] = relative_joint_ori[15, :]


# for joint, rot in zip(joint_names[1:], np.degrees(test_joint_ori)):
#     print(f"{joint} relative rot is {rot}\n")
# relative_joint_ori = test_joint_ori.reshape(1, -1)
# smplx_output = human_model(body_pose=relative_joint_ori)



# joints = smplx_output.joints
# joints = joints[:24] # remove the extra joints "The remaining 21 points are vertices selected to match some 2D keypoint annotations

  
# %%

# output_file = "viz_output/pose1.png"
# vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
# model = human_model
# joints =  output.joints.detach().cpu().numpy().squeeze()

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# # ax.view_init(elev=90., azim=-90., roll = 0)
# mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
# face_color = (1.0, 1.0, 0.9)
# edge_color = (0, 0, 0)
# mesh.set_edgecolor(edge_color)
# mesh.set_facecolor(face_color)
# ax.add_collection3d(mesh)
# # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
# joints = joints[:24, :]
# print(f'plotting joints: {joints.shape}')
# ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=1, marker='o', color='r', s=10)
# plt.savefig(output_file, dpi=200)