import importlib
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import joint_names
importlib.reload(joint_names)
from joint_names import MOCAP_JOINT_NAMES, SMPL_JOINT_NAMES

import bpy
from mathutils import Euler
import numpy as np

# ----------------- Bone realted -----------------
def get_global_bone_rot(armature_name, bone):
    """
    get global rotation of a bone
    output: global rotation of the bone
    @param armature: armature object
    @param bone: bone object that you need to get global rotation
    @return: global rotation of the bone: a rotation matrix
    """
    armature = bpy.context.scene.objects[armature_name]
    global_mat = armature.matrix_world @ bone.matrix
    return global_mat

def get_bone_global_loc(armature_name, bone_name):
    """
    Get the global location of a bone.
    @param armature_name: The name of the armature.
    @param bone_name: The name of the bone.
    @return: The global location of the bone.
    """
    armature = bpy.context.scene.objects[armature_name]
    bone = armature.pose.bones[bone_name]
    return armature.matrix_world @ bone.head

def get_all_global_bone_rot(smpl_armature_name):
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
        curr_smpl_rot.append(get_global_bone_rot(smplx_armature.name, smplx_bone).to_euler('XYZ'))
    return curr_smpl_rot

def set_global_bone_rot(armature, bone, world_rotation):
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

def set_local_bone_rot(armature_name, bone_name, local_rot):
    """
    set local rotation of a bone
    @param armature: armature name
    @param bone: bone name that you need to set local rotation
    @param local_rot: local rotation of the bone in degrees
    """
    armature = bpy.context.scene.objects[armature_name]
    bone = armature.pose.bones[bone_name] 
    # convert local_rot to euler
    local_rot = Euler(np.deg2rad(local_rot), 'XYZ')
    bone.rotation_mode = 'XYZ'
    bone.rotation_euler = local_rot
    bpy.context.view_layer.update()
    return bone


def get_all_local_bone_rot(armature_name, joint_list):
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


# ----------------- Armature related -----------------
def get_global_obj_rot(obj_name):
    """
    Get the global rotation of an object.
    @param obj_name: The name of object.
    @return: The global rotation (as Euler angles) of the object.
    """
    obj = bpy.context.scene.objects[obj_name]
    return obj.matrix_world.to_euler('XYZ')

def align_object_direction(SMPL_armature_name, MOCAP_armature_name):
    """
    Making sure two armature object are facting the same direction. Warning: the rotation is hard coded
    @param SMPL_armature_name: The name of the object that you want to align
    @param MOCAP_armature_name  : The name of the object that you want to align to
    """
    SMPL_armature_name = bpy.context.scene.objects[SMPL_armature_name]
    reference_obj = bpy.context.scene.objects[MOCAP_armature_name]
    # Get the world rotation of the reference object
    ref_world_rotation = reference_obj.matrix_world.to_euler()
    adjusted_rotation = Euler((-(ref_world_rotation.x - np.radians(90)), 
                                         -ref_world_rotation.y, 
                                         ref_world_rotation.z - np.radians(180)), 'XYZ')
    SMPL_armature_name.rotation_euler = adjusted_rotation

def get_obj_location(obj_name):
    """
    Get the location of an object.
    @param obj_name: The name of object.
    @return: The location of the object.
    """
    obj = bpy.context.scene.objects[obj_name]
    return obj.location

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
    

# ----------------- Others -----------------
# def set_ith_local_rot(armature_name, bone_idx, local_rot):
#     """
#     set local rotation of a bone
#     @param armature: armature name
#     @param bone_idx: bone index that you need to set local rotation
#     @param local_rot: local rotation of the bone
#     """
#     if armature_name.startswith('SMPL'):
#         joint_list = SMPL_JOINT_NAMES
#     else:
#         joint_list = MOCAP_JOINT_NAMES
#     bone_name = joint_list[bone_idx]
#     bone = set_local_bone_rot(armature_name, bone_name, local_rot)
#     return bone

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
        mocap_bone_global_rot = get_global_bone_rot(mocap_armature.name, mocap_bone).to_3x3()
        transformed_smpl_global_rot = mocap_bone_global_rot @ curr_trans
        # set the global rotation of smplx bone
        bone = set_global_bone_rot(smplx_armature, smplx_bone, transformed_smpl_global_rot.to_euler('XYZ'))
    
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

def set_armature_origin_to_bone_head(armature_name, bone_name):
    """
    Set the origin of an armature to the head of a bone.
    @param armature_name: The name of the armature.
    @param bone_name: The name of the bone that the origin should be set to.
    """
    # Ensure we are in object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    # Select the armature
    armature = bpy.data.objects[armature_name]
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    # Switch to edit mode to access the bone
    bpy.ops.object.mode_set(mode='EDIT')
    # Find the bone and set the cursor to its head
    edit_bone = armature.data.edit_bones[bone_name]
    bpy.context.scene.cursor.location = edit_bone.head
    # Switch back to object mode and set origin to 3D cursor
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')


def reset_blender():
    """
    Reset the blender scene
    """
    bpy.ops.wm.read_homefile(use_empty=True)

def empty_scene():
    """
    Empty the blender scene
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    bpy.ops.outliner.orphans_purge()

def find_max_frame_in_actions():
    max_frame = 0
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                max_frame = max(max_frame, keyframe.co.x)
    return max_frame