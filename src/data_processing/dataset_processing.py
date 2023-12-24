import bpy, json, re
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
FRAME_RATE = 60 # 60 fps

# ----------------- Main -----------------
dataset_dir = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_data/fbx_files"
output_dir = "/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data/preprocessed_data"

if not os.path.exists(dataset_dir):
    raise Exception(f'{dataset_dir} does not exist')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# get all .fbx files in the directory
mocap_fbx_list = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.fbx')]
mocap_fbx_list.sort()
# need for for loop to loop through all the mocap data, but for now, just use one
mocap_fbx_list = mocap_fbx_list[49:]
for mocap_fbx_path in mocap_fbx_list:
    # use regex to match the trial number and actor name
    trial_num = re.search(r'Trial_(\d+)', mocap_fbx_path).group(1)
    actor_name = re.search(r'Trial_\d+_(\w+)', mocap_fbx_path).group(1)
    print(f"Start to process {actor_name}'s data in Trial {trial_num}")

    # load the mocap armature fbx 
    # mocap_fbx = "E:\Downloads\Falling_Dataset_Session2_100-115\Falling_Dataset_Session2_100-115\Trial_100\Kate.fbx"
    curr_mocap_joint_name = [ f'{actor_name}:{name}' for name in MOCAP_JOINT_NAMES]
    bpy.ops.import_scene.fbx(filepath=mocap_fbx_path, automatic_bone_orientation = True)
    print(f'--- {actor_name} loaded ---')
    # load the smpl armature from add-on
    bpy.data.window_managers["WinMan"].smpl_tool.smpl_gender = 'female'
    bpy.ops.scene.smpl_add_gender()
    print(f'--- smpl armature loaded ---')

    # Setup up Mocap 
    # mocap_armature = bpy.context.scene.objects[f'{actor_name}:Hips']
    try:
        mocap_armature = bpy.context.scene.objects[f'{actor_name}:Hips']
    except:
        # find the armature that ends with Hips
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE' and obj.name.endswith('Hips'):
                mocap_armature_name = obj.name
                print(f'successfully found {mocap_armature_name}')
                break

        actor_name = mocap_armature_name.split(':')[0]
        mocap_armature = bpy.context.scene.objects[f'{actor_name}:Hips']
        curr_mocap_joint_name = [ f'{actor_name}:{name}' for name in MOCAP_JOINT_NAMES]
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
    print(f'--- Start to process Trial_{trial_num} ---')
    for frame in range(scene.frame_start, scene.frame_end):
        scene.frame_current = frame
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
            blender_utils.update_smpl_rotation(smpl_armature.name, mocap_armature.name, curr_mocap_joint_name, SMPL_JOINT_NAMES)
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
    save_path = os.path.join(output_dir, f'Trial_{trial_num}.csv')
    np.savetxt(save_path, dataset, delimiter=',', fmt='%s')
    print(f'--- Trial_{trial_num} saved ---')
    blender_utils.reset_blender()

