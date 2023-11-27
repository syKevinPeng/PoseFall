# %%
import numpy as np
from pathlib import Path
from icecream import ic
from scipy.datasets import face
from icecream import ic
import torch
import pandas as pd
import sys

from tqdm import tqdm

sys.path.append("../utils")
from joint_names import MOCAP_JOINT_NAMES, SMPL_JOINT_NAMES
import smplx
import pickle

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pytorch3d, torch, imageio

# visulize use pytorch3d
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    look_at_view_transform,
    HardPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex
# %%
DATAPATH = "/home/siyuan/research/PoseFall/data/processed_data/Trial_100.csv"
VIZ_OUTPUT = "/home/siyuan/research/PoseFall/src/visulization/viz_output"
DATAPATH = Path(DATAPATH)
VIZ_OUTPUT = Path(VIZ_OUTPUT)

# from smplx.joint_names import  JOINT_NAMES, SMPL_JOINT_NAMES
model_folder = "/home/siyuan/research/PoseFall/data/SMPL_cleaned"
male_model = "/home/siyuan/research/PoseFall/data/SMPL_cleaned/SMPL_MALE.pkl"
device = torch.device("cuda:0")


def load_data(path=DATAPATH):
    df = pd.read_csv(path, header=0)
    return df


# initialization
def init_human_model():
    human_model = smplx.SMPL(
        model_path=model_folder,
        create_body_pose=True,
        body_pose=None,
        create_betas=True,
        betas=None,
        gender="male",
    )
    output = human_model(return_verts=True, return_full_pose=True)
    return human_model, output


def get_motion_param(frame_num, df):
    arm_trans = df.loc[:, ["arm_loc_x", "arm_loc_y", "arm_loc_z"]]
    arm_rot = df.loc[:, ["arm_rot_x", "arm_rot_y", "arm_rot_z"]]
    joint_rot = df.loc[:, "Pelvis_x":"R_Hand_z"]
    new_pose = new_pose = torch.tensor(
        joint_rot.iloc[frame_num].values, dtype=torch.float32
    )
    new_pose = new_pose.view(-1, 3)
    bone_rot = new_pose[1:, :]

    # translation
    arm_trans = torch.tensor(arm_trans.iloc[frame_num].values, dtype=torch.float32)
    arm_trans = arm_trans.view(1, 3)

    # global orientation
    arm_rot = torch.tensor(arm_rot.iloc[frame_num].values, dtype=torch.float32)
    # arm_rot = torch.zeros_like(arm_rot)
    arm_rot = arm_rot.view(1, 3)
    return [bone_rot, arm_trans, arm_rot]


def update_model(human_model, params):
    bone_rot, arm_trans, arm_rot = params
    output = human_model(
        body_pose=bone_rot.reshape(1, -1),
        return_verts=True,
        transl=arm_trans,
        global_orient=arm_rot,
        return_full_pose=True,
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()
    return vertices, joints


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer
# Function to create grid lines
def create_ground_plane(size=2, center=(0, 0, 0), color=[0, 0, 1]):
    """
    Create a square ground plane.
    size: Length of the square side
    center: Center of the square in (x, y, z)
    color: Color of the ground plane
    """
    half_size = size / 2
    cx, cy, cz = center
    # Four corners of the square
    verts = torch.tensor([
        [cx - half_size, cy, cz - half_size],
        [cx - half_size, cy, cz + half_size],
        [cx + half_size, cy, cz + half_size],
        [cx + half_size, cy, cz - half_size]
    ], dtype=torch.float32)
    # Two triangles to form the square
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int64)
    # Color (repeated for each vertex)
    colors = torch.tensor([color for _ in range(verts.shape[0])], dtype=torch.float32)

    ground_mesh = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(verts_features=[colors]))
    return ground_mesh

# %%
# render an image
lights = pytorch3d.renderer.PointLights(location=[[0, 0, 3]], device=device)
renderer = get_mesh_renderer(image_size=512, lights=lights, device=device)
# create the grid
ground_plane = create_ground_plane().to(device)

# loop through all the frames
R, T = look_at_view_transform(dist=8.0, elev=0, azim=0)
T[0][1] -= 1
cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device, fov=60).to(
    device
)

rends = []
data = load_data()
human_model, _ = init_human_model()
max_frame = len(data)
for frame_num in tqdm(range(0, max_frame)):
    params = get_motion_param(frame_num, df=data)
    vertices, joints = update_model(human_model, params)
    vertices = torch.tensor(vertices).to(device)
    vertices = vertices.unsqueeze(0)
    faces = human_model.faces.astype(np.int64)
    textures = torch.ones_like(vertices)
    color = torch.tensor([0.7, 0.7, 1]).to(device)
    textures = textures * color
    textures = pytorch3d.renderer.TexturesVertex(textures).to(device)

    faces = torch.tensor(faces).to(device)
    faces = faces.unsqueeze(0)

    rendered_mesh = pytorch3d.structures.Meshes(
        verts=vertices, faces=faces, textures=textures
    )
    rendered_mesh = rendered_mesh.to(device)
    combined_verts = torch.cat([rendered_mesh.verts_list()[0], ground_plane.verts_list()[0]], dim=0)
    combined_faces = torch.cat([rendered_mesh.faces_list()[0], ground_plane.faces_list()[0] + rendered_mesh.verts_list()[0].shape[0]], dim=0)
    combined_textures = TexturesVertex(verts_features=torch.cat([rendered_mesh.textures.verts_features_list()[0], ground_plane.textures.verts_features_list()[0]], dim=0).unsqueeze(0))

    combined_mesh = Meshes(verts=[combined_verts], faces=[combined_faces], textures=combined_textures).to(device)

    rend = renderer(combined_mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # convert to uint8
    rend = rend * 255
    rend = rend.astype("uint8")
    rends.append(rend)


imageio.mimsave(VIZ_OUTPUT / "example.gif", rends, fps=24, loop=0)

# %%
# create a 360 view
# azims = torch.linspace(0, 360, steps=36)
# rends = []
# for i in range(len(azims)):
#     R, T = look_at_view_transform(dist=3.0, elev=0, azim=azims[i])
#     cameras = pytorch3d.renderer.FoVPerspectiveCameras(
#         R=R, T=T, device=device, fov=60
#     ).to(device)
#     rend = renderer(jointed_mesh, cameras=cameras, lights=lights)
#     rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
#     # convert to uint8
#     rend = rend * 255
#     rends.append(rend.astype("uint8"))
