"""
Based on generated human pose parameter, we reconstruct the human motion in blender
"""
import bpy, json
import numpy as np
import importlib
import sys, os
import math
from mathutils import Euler, Matrix

# joint_names
SMPL_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

def empty_scene():
    """
    Empty the blender scene
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    bpy.ops.outliner.orphans_purge()
    
def set_obj_location(obj_name, location):
    """
    Set the location of an object.
    @param obj_name: The name of object.
    @param location: The location to set.
    """
    obj = bpy.context.scene.objects[obj_name]
    obj.location = location
    bpy.context.view_layer.update()

def set_obj_rotation(obj_name, rotation):
    """
    Set the rotation of an object.
    @param obj_name: The name of object.
    @param rotation: The rotation to set.
    """
    obj = bpy.context.scene.objects[obj_name]
    obj.rotation_euler = rotation
    bpy.context.view_layer.update()

def set_all_local_bone_rot(armature_name, joint_list, joint_local_rot):
    '''
    Set the local rotations of all bones in an SMPLX armature.

    @param armature_name: The name of the armature.
    @param joint_list: A list of joint names.
    @param joint_local_rot: A list of local rotations (as Euler angles) for each bone in the SMPLX armature.
    '''
    # get the current smpl bone rotation
    armature = bpy.context.scene.objects[armature_name]
    for bone_name, local_rot in zip(joint_list, joint_local_rot):
        bone = armature.pose.bones[bone_name]
        # set rotation mode of the bone to euler:
        bone.rotation_mode = 'XYZ'
        local_rot_mat = Euler(local_rot, 'XYZ').to_matrix()
        bone.rotation_euler = local_rot_mat.to_euler('XYZ')
    # Update the view layer to reflect changes
    bpy.context.view_layer.update()


# remove everything
empty_scene()
# Taking cmd args
argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_csv_path = str(argv[0])
output_directory = str(argv[1])
print(f'processing input csv file: {input_csv_path}')


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
bpy.context.scene.camera = camera_obj

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

