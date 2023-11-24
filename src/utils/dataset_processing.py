from time import sleep
import bpy
import numpy as np
from mathutils import Euler, Matrix
# Mocap Joint Names
MOCAP_JOINT_NAMES = [
    "Spine",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine1",
    "LeftLeg",
    "RightLeg",
    "Spine2",
    "LeftFoot",
    "RightFoot",
    "Spine3",
    "LeftToeBase",
    "RightToeBase",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    "LeftHandEE",
    "RightHandEE",
]
# SMPL Joint Names
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

def get_global_rot(armature, bone):
    """
    get global rotation of a bone
    output: global rotation of the bone
    @param armature: armature object
    @param bone: bone object that you need to get global rotation
    """
    global_mat = armature.matrix_world @ bone.matrix
    return global_mat

def calculate_bone_trans_matrix(mocap_armature_name, smplx_armature_name, mocap_bone_name, smplx_bone_name):
    """
    calculate the ration matrix needed to aligh the rotation of mocap bone to smplx bone
    @param mocap_armature_name: mocap armature name
    @param smplx_armature_name: smplx armature name
    @param mocap_bone_name: mocap bone name
    @param smplx_bone_name: smplx bone name
    """
    mocap_armature = bpy.context.scene.objects[mocap_armature_name]
    smplx_armature = bpy.context.scene.objects[smplx_armature_name]
    mocap_bone = mocap_armature.pose.bones[mocap_bone_name]
    smplx_bone = smplx_armature.pose.bones[smplx_bone_name]
    # get the bone'sd rest pose matrices in global space
    mocap_bone_rest_matrix_world = mocap_armature.matrix_world @ mocap_bone.bone.matrix_local
    smplx_bone_rest_matrix_world = smplx_armature.matrix_world @ smplx_bone.bone.matrix_local
    # conver tot rotation matrix
    mocap_bone_rest_rot_matrix_world = mocap_bone_rest_matrix_world.to_3x3()
    smplx_bone_rest_rot_matrix_world = smplx_bone_rest_matrix_world.to_3x3()
    # calculate the inverse of bone mocap's global rotation matrix
    mocap_bone_rest_rot_matrix_world_inv = mocap_bone_rest_rot_matrix_world.inverted()
    # calculate the rotation matrix from mocap bone to smplx bone
    mocap_to_smplx_rot_matrix = mocap_bone_rest_rot_matrix_world_inv @ smplx_bone_rest_rot_matrix_world
    return mocap_to_smplx_rot_matrix


def calculate_bone_trans_matrix(mocap_armature_name, smplx_armature_name, mocap_bone_name, smplx_bone_name):
    """
    calculate the ration matrix needed to aligh the rotation of mocap bone to smplx bone
    @param mocap_armature_name: mocap armature name
    @param smplx_armature_name: smplx armature name
    @param mocap_bone_name: mocap bone name
    @param smplx_bone_name: smplx bone name
    """
    mocap_armature = bpy.context.scene.objects[mocap_armature_name]
    smplx_armature = bpy.context.scene.objects[smplx_armature_name]
    mocap_bone = mocap_armature.pose.bones[mocap_bone_name]
    smplx_bone = smplx_armature.pose.bones[smplx_bone_name]
    # get the bone'sd rest pose matrices in global space
    mocap_bone_rest_matrix_world = mocap_armature.matrix_world @ mocap_bone.bone.matrix_local
    smplx_bone_rest_matrix_world = smplx_armature.matrix_world @ smplx_bone.bone.matrix_local
    # conver tot rotation matrix
    mocap_bone_rest_rot_matrix_world = mocap_bone_rest_matrix_world.to_3x3()
    smplx_bone_rest_rot_matrix_world = smplx_bone_rest_matrix_world.to_3x3()
    # calculate the inverse of bone mocap's global rotation matrix
    mocap_bone_rest_rot_matrix_world_inv = mocap_bone_rest_rot_matrix_world.inverted()
    # calculate the rotation matrix from mocap bone to smplx bone
    mocap_to_smplx_rot_matrix = mocap_bone_rest_rot_matrix_world_inv @ smplx_bone_rest_rot_matrix_world
    return mocap_to_smplx_rot_matrix

def set_all_global_armature_rot(armature, bone, world_rotation):
    """
    Set the global rotation of a bone in an armature.
    @param armature: The armature object.
    @param bone: The bone object.
    @param world_rotation: The global rotation (as Euler angles) to set.
    """
    # set rotation mode of the bone to euler:
    bone.rotation_mode = 'XYZ'
    if bone.parent:
        # Get the parent's world transformation matrix
        parent_matrix_world = armature.matrix_world @ bone.parent.matrix
        # Invert the parent's world matrix to transform back into its local space
        parent_matrix_world_inv = parent_matrix_world.inverted()
        # Convert the world rotation to the parent's local space
        bone_rotation_local = parent_matrix_world_inv @ world_rotation.to_matrix().to_4x4()
        # Now, bone_rotation_local is a 4x4 transformation matrix in the parent's local space
        # To apply this to the bone, you need to extract the rotation component
        local_rotation = bone_rotation_local.to_euler("XYZ")
        # print(f'relative rotation for bone {bone}: {local_rotation}')
        # Apply this local rotation to the bone
        bone.rotation_euler = local_rotation
    else:
        # If the bone has no parent, then its local space is the same as its world space
        # So we can just apply the world rotation directly
        bone.rotation_euler = world_rotation
    bpy.context.view_layer.update()
    return bone

def get_all_global_armature_rot(smpl_armature_name):
    """
    Get the current global rotations of all bones in an SMPLX armature.

    @param smpl_armature_name: The name of the armature.
    @return: A list of global rotations (as Euler angles) for each bone in the SMPLX armature.
    """
    # get the current smpl bone rotation
    smplx_armature = bpy.context.scene.objects[smpl_armature_name]
    curr_smpl_rot = []
    for smplx_bone_name in SMPL_JOINT_NAMES:
        smplx_bone = smplx_armature.pose.bones[smplx_bone_name]
        curr_smpl_rot.append(get_global_rot(smplx_armature, smplx_bone).to_euler('XYZ'))
    return curr_smpl_rot

    
def get_all_local_armature_rot(armature_name, joint_list):
    '''
    Get the current local rotations of all bones in an SMPLX armature.

    @param armature_name: The name of the armature.
    @param joint_list: A list of joint names.
    @return: A list of local rotations (as Euler angles) for each bone in the SMPLX armature.
    '''
    # get the current smpl bone rotation
    armature = bpy.context.scene.objects[armature_name]
    joint_local_rot = []
    for bone_name in joint_list:
        bone = armature.pose.bones[bone_name]
        bone.rotation_mode = 'XYZ'
        local_rot_euler = bone.rotation_euler
        joint_local_rot.append(local_rot_euler)
    return joint_local_rot

def set_all_local_armature_rot(armature_name, joint_list, joint_local_rot):
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

def update_smpl_rotation(smpl_armature_name, mocap_armature_name, mocap_joint_names, smpl_joint_names):
    # Setup up Mocap 
    mocap_armature = bpy.context.scene.objects[mocap_armature_name]
    # setup up Smplex
    smplx_armature = bpy.context.scene.objects[smpl_armature_name]
    # calculate for all bones:
    for mocap_bone_name, smplx_bone_name in zip(mocap_joint_names, smpl_joint_names):
        mocap_bone = mocap_armature.pose.bones[mocap_bone_name]
        smplx_bone = smplx_armature.pose.bones[smplx_bone_name]
        # get the global transformation matrix of mocap bone
        curr_trans = calculate_bone_trans_matrix(mocap_armature.name, smplx_armature.name, mocap_bone.name, smplx_bone.name)
        # get the global rotation matrix of mocap bone
        mocap_bone_global_rot = get_global_rot(mocap_armature, mocap_bone).to_3x3()
        transformed_smpl_global_rot = mocap_bone_global_rot @ curr_trans
        # set the global rotation of smplx bone
        bone = set_all_global_armature_rot(smplx_armature, smplx_bone, transformed_smpl_global_rot.to_euler('XYZ'))
    
def reset_armature(armature_name):
    """
    Reset the rotation of all bones in an armature.
    @param armature_name: The name of the armature.
    """
    # set all bones's rotation to zero
    mocap=bpy.context.scene.objects[armature_name]
    for bone in mocap.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = (0,0,0)

def reset_blender():
    """
    Reset the blender scene
    """
    bpy.ops.wm.read_homefile(use_empty=True)



# ----------------- Main -----------------
scene = bpy.data.scenes['Scene']
# load the mocap armature fbx
mocap_fbx = "E:\Downloads\Falling_Dataset_Session2_100-115\Falling_Dataset_Session2_100-115\Trial_100\Kate.fbx"
actor_name = mocap_fbx.split('\\')[-1].split('.')[0]
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
smplx_armature = bpy.context.scene.objects['SMPL-female']
# put smplx on the ground
bpy.ops.object.smpl_snap_ground_plane()


# loop through all the frames
for frame in range(1, scene.frame_end):
    scene.frame_current = frame
    print(f'--- processing frame {frame} ---')
    updated_flag = False
    curr_smplx_param = np.array(get_all_global_armature_rot(smplx_armature.name))
    while(not updated_flag):
        print(f'looping-----')
        update_smpl_rotation(smplx_armature.name, mocap_armature.name, MOCAP_JOINT_NAMES, SMPL_JOINT_NAMES)
        updated_smplx_param = np.array(get_all_global_armature_rot(smplx_armature.name))
        print(f'curr_smplx_param: {curr_smplx_param}')
        print(f'updated_smplx_param: {updated_smplx_param}')
        if np.array_equal(updated_smplx_param, curr_smplx_param):
            updated_flag = True
        else:
            curr_smplx_param = updated_smplx_param

    break

local_rot = get_all_local_armature_rot(mocap_armature.name, MOCAP_JOINT_NAMES)
local_rot = np.array(local_rot)
# save local_rot
np.save(f'{actor_name}_local_rot.npy', local_rot)