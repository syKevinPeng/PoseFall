"""
Based on generated human pose parameter, we reconstruct the human motion in blender
"""
import bpy, json
import numpy as np
import importlib
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import blender_utils
importlib.reload(blender_utils)
from blender_utils import *
import joint_names
importlib.reload(joint_names)
from joint_names import MOCAP_JOINT_NAMES, SMPL_JOINT_NAMES

# remove everything
empty_scene()

trial_num = "Trial_100"
# output_dir = "../../data/processed_data"
output_dir = "./"
save_path = os.path.join(output_dir, f'{trial_num}.csv')

# Create a new SMPL armature and let it move acccording to recorded data
bpy.data.window_managers["WinMan"].smpl_tool.smpl_gender = 'female'
bpy.ops.scene.smpl_add_gender()
smpl_armature = bpy.context.scene.objects['SMPL-female']

# from each frame, read the data and set the armature's rotation, location
# load the dataset
dataset = np.loadtxt(save_path, delimiter=',', dtype=str)
column_name = dataset[0]
dataset = dataset[1:]


pre_aug_start = 35
pre_aug_end = 55
post_aug_start = 95
post_aug_end = 105
from scipy.interpolate import CubicSpline

# select mid 50 frames
aug_dataset = dataset[50:100]
# perform fft along the time dimension
fft_data = np.fft.fft(aug_dataset[:, 7:].astype(float), axis=0)
scale_factor = 0.8
magnitudes = np.abs(fft_data)*scale_factor
phases = np.angle(fft_data)
scaled_fft_data = magnitudes * np.exp(1j*phases)
# inverse fft
sacled_bone_rot = np.fft.ifft(scaled_fft_data, axis=0)
dataset[50:100, 7:] = sacled_bone_rot.real.astype(float)

# applying cubic spline interpolation
# Create spline interpolation for the start boundary
x = np.arange(pre_aug_start, pre_aug_end)
y_start = dataset[x, 7:].astype(float)
cs_start = CubicSpline(x, y_start, axis=0, bc_type='natural')

# Create spline interpolation for the end boundary
x = np.arange(post_aug_start, post_aug_end)
y_end = dataset[x, 7:].astype(float)
cs_end = CubicSpline(x, y_end, axis=0, bc_type='natural')

# Apply the interpolation to create a smooth transition
interp_start = cs_start(np.linspace(pre_aug_start, pre_aug_end, pre_aug_end - pre_aug_start))
interp_end = cs_end(np.linspace(post_aug_start, post_aug_end, post_aug_end - post_aug_start))

# Replace the boundary regions
dataset[pre_aug_start:pre_aug_end, 7:] = interp_start
dataset[post_aug_start:post_aug_end, 7:] = interp_end


# loop through each frame
for frame in dataset:
    # convert frame number from string to int
    frame_num = frame[0].astype(float)
    frame_num  = int(frame_num)
    # set the armature's object rotation
    arm_rot = frame[1:4].astype(float)
    # arm_rot = Euler(arm_rot, 'XYZ')
    set_obj_rotation(smpl_armature.name, arm_rot)
    # set the armature's object location
    arm_loc = frame[4:7].astype(float)
    set_obj_location(smpl_armature.name, arm_loc)
    # set the armature's bone rotation
    bone_rot = frame[7:].astype(float)
    bone_rot = bone_rot.reshape(-1, 3)
    set_all_local_bone_rot(smpl_armature.name, SMPL_JOINT_NAMES, bone_rot)
    # insert key frames here:
    smpl_armature.keyframe_insert(data_path="location", frame=frame_num)
    smpl_armature.keyframe_insert(data_path="rotation_euler", frame=frame_num)
    for bone_name in SMPL_JOINT_NAMES:
        bone = smpl_armature.pose.bones[bone_name]
        bone.keyframe_insert(data_path="rotation_euler", frame=frame_num)

# # set the original armature
# bpy.ops.scene.smpl_add_gender()
# smpl_armature_ori = bpy.context.scene.objects['SMPL-female.001']
# dataset = np.loadtxt(save_path, delimiter=',', dtype=str)
# column_name = dataset[0]
# dataset = dataset[1:]
# for frame in dataset:
#     # convert frame number from string to int
#     frame_num = frame[0].astype(float)
#     frame_num  = int(frame_num)
#     # set the armature's object rotation
#     arm_rot = frame[1:4].astype(float)
#     # arm_rot = Euler(arm_rot, 'XYZ')
#     set_obj_rotation(smpl_armature_ori.name, arm_rot)
#     # set the armature's object location
#     arm_loc = frame[4:7].astype(float)
#     set_obj_location(smpl_armature_ori.name, arm_loc)
#     # set the armature's bone rotation
#     bone_rot = frame[7:].astype(float)
#     bone_rot = bone_rot.reshape(-1, 3)
#     set_all_local_bone_rot(smpl_armature_ori.name, SMPL_JOINT_NAMES, bone_rot)
#     # insert key frames here:
#     smpl_armature_ori.keyframe_insert(data_path="location", frame=frame_num)
#     smpl_armature_ori.keyframe_insert(data_path="rotation_euler", frame=frame_num)
#     for bone_name in SMPL_JOINT_NAMES:
#         bone = smpl_armature_ori.pose.bones[bone_name]
#         bone.keyframe_insert(data_path="rotation_euler", frame=frame_num)