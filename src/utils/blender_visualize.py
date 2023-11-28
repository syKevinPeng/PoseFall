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