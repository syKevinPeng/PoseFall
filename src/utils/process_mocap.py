import sys
import bpy
import csv
import argparse


Captury_boens = [
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

# Convert the global rotation to local rotation (with respect to the parent bone)
# input: blender bone object
# output: relative rotation matrix
def convert_global_rot_to_local_rot(bone):
    if bone.parent:
        return bone.parent.matrix.inverted() @ bone.matrix
    else:
        return bone.matrix


def process_mocap_data(fbx_file_path, armature_name, output_file):
    # Import the FBX file
    bpy.ops.import_scene.fbx(filepath=fbx_file_path,  automatic_bone_orientation=True)

    # Ensure the armature is present in the scene
    if armature_name not in bpy.context.scene.objects:
        raise ValueError("Armature not found in the scene. Please check the name.")

    # Select the armature
    armature = bpy.context.scene.objects[armature_name]
    armature_prefix = armature_name.split(':')[0]

    # Get the scene for frame manipulation
    scene = bpy.context.scene

    # Start and end frame of your animation
    start_frame = scene.frame_start
    end_frame = scene.frame_end

    # Open the file with CSV writer
    with open(output_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        # Write the header
        header = ['Frame']
        for bone in armature.pose.bones:
            bone_name = bone.name.split(':')[-1]
            header += [bone_name + '_X', bone_name + '_Y', bone_name + '_Z']
        csv_writer.writerow(header)

        # Iterate over each frame
        for frame in range(start_frame, end_frame + 1):
            # Set the scene's current frame
            scene.frame_set(frame)

            # Write the frame number
            row = [frame]

            # Iterate over each bone in the armature
            for bone_name in Captury_boens:
                bone_name = armature_prefix + ':' + bone_name
                # get the boen in armature
                bone = armature.pose.bones[bone_name]
                # Get the local rotation 
                local_rot_mat = convert_global_rot_to_local_rot(bone)
                euler_rotation = local_rot_mat.to_euler()

                # Append the rotation to the row
                row += [euler_rotation.x, euler_rotation.y, euler_rotation.z]
                # print(row)

            # Write the row to the CSV file
            csv_writer.writerow(row)

    print(f"Bone rotations written to {output_file}")

fbx_file_path = "/home/siyuan/research/PoseFall/data/MoCap/Siyuan.fbx"
armature_name = "Siyuan:Hips"
output_file = "bone_rotations.csv"

process_mocap_data(fbx_file_path, armature_name, output_file)

# if __name__ == '__main__':
#     # Create the parser
#     parser = argparse.ArgumentParser(description='Process mocap data.')

#     # Add the arguments
#     parser.add_argument('--fbx_file_path', type=str, help='The path to the FBX file', default= "/home/siyuan/research/PoseFall/data/MoCap/Siyuan.fbx")
#     parser.add_argument('--armature_name', type=str, help='The name of the armature', default= "Siyuan:Hips")
#     parser.add_argument('--output_file', type=str, help='The name of the output file', default= "bone_rotations.csv")

#     # Parse the arguments
#     args = parser.parse_args()

#     # Call the function with the command line arguments
#     process_mocap_data(args.fbx_file_path, args.armature_name, args.output_file)
