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

from scipy.interpolate import interp1d

# select mid 50 frames
aug_dataset = dataset
print(f"Before size: {aug_dataset.shape}")
speed_factor = 5
time_vector = np.arange(aug_dataset.shape[0])
new_time_vector = np.linspace(time_vector[0], time_vector[-1], int(len(time_vector) / speed_factor))
interpolating_function = interp1d(time_vector, aug_dataset, axis=0, kind='linear')
resampled_data = interpolating_function(new_time_vector)
# perform fft along the time dimension
fft_data = np.fft.fft(resampled_data.astype(float), axis=0)
scale_factor = 0.95
magnitudes = np.abs(fft_data)*scale_factor
phases = np.angle(fft_data)
scaled_fft_data = magnitudes * np.exp(1j*phases)
# inverse fft
sacled_bone_rot = np.fft.ifft(scaled_fft_data, axis=0)
dataset = sacled_bone_rot.real.astype(float)
print(dataset.shape)

# add randomly z-axis rotation
random_rotation = np.random.uniform(-np.pi/2, np.pi/2)
# get the rotations
rotations = dataset[:, 3].astype(float)
rotations += random_rotation
rotations = rotations % (2 * np.pi)
dataset[:, 3] = rotations


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