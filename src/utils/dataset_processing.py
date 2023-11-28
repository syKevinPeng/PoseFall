import bpy, json
import numpy as np
import importlib
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import blender_utils
importlib.reload(blender_utils)
import joint_names
importlib.reload(joint_names)
from joint_names import MOCAP_JOINT_NAMES, SMPL_JOINT_NAMES


# ----------------- Main -----------------
trial_num = "Trial_100"
# output_dir = "../../data/processed_data"
output_dir = "./"
# load the mocap armature fbx 
# mocap_fbx = "E:\Downloads\Falling_Dataset_Session2_100-115\Falling_Dataset_Session2_100-115\Trial_100\Kate.fbx"
mocap_fbx = "/home/siyuan/research/PoseFall/data/MoCap/Kate.fbx"
actor_name = mocap_fbx.split('/')[-1].split('.')[0]
print(f'--- loading {actor_name} ---')
MOCAP_JOINT_NAMES = [ f'{actor_name}:{name}' for name in MOCAP_JOINT_NAMES]

bpy.ops.import_scene.fbx(filepath=mocap_fbx, automatic_bone_orientation = True)
print(f'--- {actor_name} loaded ---')
# load the smpl armature from add-on
bpy.data.window_managers["WinMan"].smpl_tool.smpl_gender = 'female'
bpy.ops.scene.smpl_add_gender()
print(f'--- smpl armature loaded ---')

# Setup up Mocap 
mocap_armature = bpy.context.scene.objects[f'{actor_name}:Hips']
# setup up Smplex
smpl_armature = bpy.context.scene.objects['SMPL-female']
# move the origin of the smpl armature to the hips
blender_utils.set_armature_origin_to_bone_head(smpl_armature.name, 'Pelvis')
max_frame = None
dataset = []
# loop through all the frames
scene = bpy.data.scenes['Scene']
largest_frame = int(blender_utils.find_max_frame_in_actions())
scene.frame_end = largest_frame
scene.frame_start = 1
for frame in range(scene.frame_start, scene.frame_end):
    scene.frame_current = frame
    print(f'--- processing frame {frame} ---')
    updated_flag = False
    # set them to the same location
    blender_utils.set_obj_location(smpl_armature.name, blender_utils.get_obj_location(mocap_armature.name))
    blender_utils.set_obj_location(smpl_armature.name, blender_utils.get_obj_location(mocap_armature.name)) # I can't figure out why I need to do this twice
    # make sure they have the same orientation:
    blender_utils.align_object_direction(smpl_armature.name, mocap_armature.name)
    blender_utils.align_object_direction(smpl_armature.name, mocap_armature.name)
    # then process the relative rotation for each joint
    curr_smpl_param = np.array(blender_utils.get_all_global_bone_rot(smpl_armature.name))
    # Insert keyframes for armature object's location and rotation
    smpl_armature.keyframe_insert(data_path="location", frame=frame)
    smpl_armature.keyframe_insert(data_path="rotation_euler", frame=frame)
    while(not updated_flag):
        print(f'looping-----')
        blender_utils.update_smpl_rotation(smpl_armature.name, mocap_armature.name, MOCAP_JOINT_NAMES, SMPL_JOINT_NAMES)
        updated_smpl_param = np.array(blender_utils.get_all_global_bone_rot(smpl_armature.name))
        if np.array_equal(updated_smpl_param, curr_smpl_param):
            updated_flag = True
        else:
            curr_smpl_param = updated_smpl_param
    # record data
    # armature's object rotation
    obj_rot = blender_utils.get_global_obj_rot(smpl_armature.name)
    # obj_rot = get_global_bone_rot(smpl_armature.name, smpl_armature.pose.bones['Spine1']).to_euler('XYZ')
    # armature's object location: the global location of the root bone
    # obj_loc = get_global_bone_rot(smpl_armature.name, smpl_armature.pose.bones['root']).to_euler('XYZ')
    # get the location of the armature object
    obj_loc = blender_utils.get_obj_location(smpl_armature.name)
    # armature bone's local rotation: 
    bone_rot = blender_utils.get_all_local_bone_rot(smpl_armature.name, SMPL_JOINT_NAMES)
    
    arm_rot = np.array(obj_rot).flatten()
    arm_loc = np.array(obj_loc).flatten()
    bone_rot = np.array(bone_rot).flatten()
    # expand each joint's name into JOINT_x, JOINT_y, JOINT_z

    data = [int(frame), *arm_rot, *arm_loc, *bone_rot]
    dataset.append(data)

    # insert key frames here:
    for bone_name in SMPL_JOINT_NAMES:
        bone = smpl_armature.pose.bones[bone_name]
        bone.keyframe_insert(data_path="rotation_euler", frame=frame)

    if max_frame and frame == max_frame:
        break
    

column_name = ['frame', 'arm_rot_x', 'arm_rot_y', 'arm_rot_z', 'arm_loc_x', 'arm_loc_y', 'arm_loc_z']
for joint_name in SMPL_JOINT_NAMES:
    column_name.append(f'{joint_name}_x')
    column_name.append(f'{joint_name}_y')
    column_name.append(f'{joint_name}_z')
# append column name into the dataset
dataset = np.array(dataset)
dataset = np.vstack((column_name, dataset))

# save the dataset as csv
save_path = os.path.join(output_dir, f'{trial_num}.csv')
np.savetxt(save_path, dataset, delimiter=',', fmt='%s')


# # remove everything
# empty_scene()
# # Create a new SMPL armature and let it move acccording to recorded data
# bpy.data.window_managers["WinMan"].smpl_tool.smpl_gender = 'female'
# bpy.ops.scene.smpl_add_gender()
# smpl_armature = bpy.context.scene.objects['SMPL-female']

# # from each frame, read the data and set the armature's rotation, location
# # load the dataset
# dataset = np.loadtxt(save_path, delimiter=',', dtype=str)
# column_name = dataset[0]
# dataset = dataset[1:]
# # loop through each frame
# for frame in dataset:
#     # convert frame number from string to int
#     frame_num = frame[0].astype(float)
#     frame_num  = int(frame_num)
#     # set the armature's object rotation
#     arm_rot = frame[1:4].astype(float)
#     # arm_rot = Euler(arm_rot, 'XYZ')
#     set_obj_rotation(smpl_armature.name, arm_rot)
#     # set the armature's object location
#     arm_loc = frame[4:7].astype(float)
#     set_obj_location(smpl_armature.name, arm_loc)
#     # set the armature's bone rotation
#     bone_rot = frame[7:].astype(float)
#     bone_rot = bone_rot.reshape(-1, 3)
#     set_all_local_bone_rot(smpl_armature.name, SMPL_JOINT_NAMES, bone_rot)
#     # insert key frames here:
#     smpl_armature.keyframe_insert(data_path="location", frame=frame_num)
#     smpl_armature.keyframe_insert(data_path="rotation_euler", frame=frame_num)
#     for bone_name in SMPL_JOINT_NAMES:
#         bone = smpl_armature.pose.bones[bone_name]
#         bone.keyframe_insert(data_path="rotation_euler", frame=frame_num)