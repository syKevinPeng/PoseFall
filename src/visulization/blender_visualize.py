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
import math
# remove everything
empty_scene()


# Taking cmd args
input_csv_path = str(sys.argv[1])
output_directory = str(sys.argv[2])



# Create a new SMPL armature and let it move acccording to recorded data
bpy.data.window_managers["WinMan"].smpl_tool.smpl_gender = 'female'
bpy.ops.scene.smpl_add_gender()
smpl_armature = bpy.context.scene.objects['SMPL-female']

# from each frame, read the data and set the armature's rotation, location
# load the dataset
dataset = np.loadtxt(input_csv_path, delimiter=',', dtype=str)
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
        
        
# add camera
camera_data = bpy.data.cameras.new(name = "camera")
camera_obj = bpy.data.objects.new("camera", camera_data)
bpy.context.collection.objects.link(camera_obj)

# Position the camera
camera_obj.location = (7, 0, -0.5)
camera_obj.rotation_euler = np.radians((90, 0, 90))

# add lights
# Create the light data
light_data = bpy.data.lights.new(name='MyLight', type='POINT')

# Create the light object
light_object = bpy.data.objects.new(name='MyLight', object_data=light_data)

# Add the light to the scene
bpy.context.collection.objects.link(light_object)
light_object.location = (7, 1, 0)
# Position the light
light_data.energy = 1000 

# Save it as PNG files
# Specify the directory to save the PNG files
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Set render settings
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Render each frame and save as PNG
for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    bpy.context.scene.render.filepath = os.path.join(output_directory, f"frame_{frame:03d}.png")
    bpy.ops.render.render(write_still=True)

