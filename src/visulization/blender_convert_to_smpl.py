import bpy
import numpy as np
# Mocap Joint Names
MOCAP_JOINT_NAME = [
    # "Spine",
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
    # "pelvis",
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
    # '''
    # get global rotation of a bone
    # output: global rotation of the bone
    # @param armature: armature object
    # @param bone: bone object that you need to get global rotation
    # '''
    global_mat = armature.matrix_world @ bone.matrix
    return global_mat

def calculate_bone_trans_matrix(mocap_armature_name, smplx_armature_name, mocap_bone_name, smplx_bone_name):
    # '''
    # calculate the ration matrix needed to aligh the rotation of mocap bone to smplx bone
    # @param mocap_armature_name: mocap armature name
    # @param smplx_armature_name: smplx armature name
    # @param mocap_bone_name: mocap bone name
    # @param smplx_bone_name: smplx bone name
    # '''
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
    # '''
    # calculate the ration matrix needed to aligh the rotation of mocap bone to smplx bone
    # @param mocap_armature_name: mocap armature name
    # @param smplx_armature_name: smplx armature name
    # @param mocap_bone_name: mocap bone name
    # @param smplx_bone_name: smplx bone name
    # '''
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

def set_bone_to_global_orientation(armature, bone, world_rotation):
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
        set_bone_to_global_orientation(smplx_armature, smplx_bone, transformed_smpl_global_rot.to_euler('XYZ'))

def reset_armature(armature_name):
    # set all bones's rotation to zero
    mocap=bpy.context.scene.objects[armature_name]
    for bone in mocap.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = (0,0,0)

def get_curr_smpl_rot(smpl_armature_name):
    # get the current smpl bone rotation
    smplx_armature = bpy.context.scene.objects[smpl_armature_name]
    curr_smpl_rot = []
    for smplx_bone_name in SMPL_JOINT_NAMES:
        smplx_bone = smplx_armature.pose.bones[smplx_bone_name]
        curr_smpl_rot.append(get_global_rot(smplx_armature, smplx_bone).to_euler('XYZ'))
    return curr_smpl_rot

def reset_blender():
    bpy.ops.wm.read_homefile(use_empty=True)

scene = bpy.data.scenes['Scene']
# load the mocap armature fbx
mocap_fbx = "E:\Downloads\Siyuan.fbx"
actor_name = mocap_fbx.split('\\')[-1].split('.')[0]
print(f'--- loading {actor_name} ---')
MOCAP_JOINT_NAME = [ f'{actor_name}:{name}' for name in MOCAP_JOINT_NAME]

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

# loop through all the frames
for frame in range(1, scene.frame_end):
    scene.frame_current = frame
    print(f'--- processing frame {frame} ---')
    updated_flag = False
    curr_smplx_param = np.array(get_curr_smpl_rot(smplx_armature.name))
    while(not updated_flag):
        print(f'looping-----')
        update_smpl_rotation(smplx_armature.name, mocap_armature.name, MOCAP_JOINT_NAME, SMPL_JOINT_NAMES)
        updated_smplx_param = np.array(get_curr_smpl_rot(smplx_armature.name))
        print(f'curr_smplx_param: {curr_smplx_param}')
        print(f'updated_smplx_param: {updated_smplx_param}')
        if np.array_equal(updated_smplx_param, curr_smplx_param):
            updated_flag = True
        else:
            curr_smplx_param = updated_smplx_param

    break

