# %%
import numpy as np
from pathlib import Path
from icecream import ic
from scipy.datasets import face
from icecream import ic
import torch
import pandas as pd
import sys
sys.path.append('../utils')
from joint_names import MOCAP_JOINT_NAMES, SMPL_JOINT_NAMES
import smplx
import pickle
DATAPATH = '/home/siyuan/research/PoseFall/data/processed_data/Trial_100.csv'
VIZ_OUTPUT = '/home/siyuan/research/PoseFall/src/visulization/viz_output'
DATAPATH = Path(DATAPATH)
VIZ_OUTPUT = Path(VIZ_OUTPUT)

# from smplx.joint_names import  JOINT_NAMES, SMPL_JOINT_NAMES 
model_folder= '/home/siyuan/research/PoseFall/data/SMPL_cleaned'
male_model = "/home/siyuan/research/PoseFall/data/SMPL_cleaned/SMPL_MALE.pkl"


import smplx
model_folder= '/home/siyuan/research/PoseFall/data/SMPL_cleaned'
human_model = smplx.SMPL(
    model_path = model_folder,
    create_body_pose=True,
    body_pose= None,
    create_betas=True,
    betas=None,
)
# get mesh and joints from the SMPL model
mesh = human_model()
joints = mesh.joints

# processing dataframe
df = pd.read_csv(DATAPATH, header=0)
arm_trans = df.loc[:,['arm_loc_x', 'arm_loc_y', 'arm_loc_z']]
arm_rot = df.loc[:,['arm_rot_x', 'arm_rot_y', 'arm_rot_z']]
joint_rot = df.loc[:,'Pelvis_x':'R_Hand_z']

# %%
import pytorch3d, torch, imageio
# visulize use pytorch3d
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    look_at_view_transform,
    HardPhongShader,
)
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
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


device = torch.device("cuda:0")
vertices = mesh.vertices
vertices = vertices.detach().cpu()
faces = human_model.faces.astype(np.int64)
textures = torch.ones_like(vertices)
color = [0.7, 0.7, 1]
textures = textures * torch.tensor(color)
textures = pytorch3d.renderer.TexturesVertex(textures).to(device)

faces = torch.tensor(faces).to(device)
faces = faces.unsqueeze(0)
vertices = torch.tensor(vertices).to(device)

viz_mesh = pytorch3d.structures.Meshes(
    verts=vertices, 
    faces=faces, 
    textures=textures)
viz_mesh = viz_mesh.to(device)
# %%
# render an xyz world axis
vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)  
faces = torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.int64) 
textures = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)  
axis_textures = pytorch3d.renderer.TexturesVertex([textures]) 
ic(vertices.shape)
ic(faces.shape)
axis_mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=axis_textures)
axis_mesh = axis_mesh.to(device)

jointed_mesh = pytorch3d.structures.join_meshes_as_batch([viz_mesh, axis_mesh])
# %%

# render an image
lights = pytorch3d.renderer.PointLights(location=[[0, 0, 3]], device=device)
renderer = get_mesh_renderer(image_size=512, lights=lights, device=device)

# create a 360 view
azims = torch.linspace(0, 360, steps=36)
rends = []
for i in range(len(azims)):
    R, T = look_at_view_transform(dist=3.0, elev=0, azim=azims[i])
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, device=device, fov=60
    ).to(device)
    rend = renderer(jointed_mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # convert to uint8
    rend = rend * 255
    rends.append(rend.astype("uint8"))


imageio.mimsave(VIZ_OUTPUT/"example.gif", rends, fps=24, loop=0)


# %%
joints.shape


def viz_360(output_path, human_model, mesh):

    device = torch.device("cuda:0")
    vertices = mesh.vertices
    vertices = vertices.detach().cpu()
    faces = human_model.faces.astype(np.int64)
    textures = torch.ones_like(vertices)
    color = [0.7, 0.7, 1]
    textures = textures * torch.tensor(color)
    textures = pytorch3d.renderer.TexturesVertex(textures).to(device)

    faces = torch.tensor(faces).to(device)
    faces = faces.unsqueeze(0)
    vertices = torch.tensor(vertices).to(device)
    viz_mesh = pytorch3d.structures.Meshes(
        verts=vertices, 
        faces=faces, 
        textures=textures)
    viz_mesh = viz_mesh.to(device)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, 3]], device=device)
    renderer = get_mesh_renderer(image_size=512, lights=lights, device=device)

    # create a 360 view
    azims = torch.linspace(0, 360, steps=36)
    rends = []
    for i in range(len(azims)):
        R, T = look_at_view_transform(dist=3.0, elev=0, azim=azims[i])
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, device=device, fov=60
        ).to(device)
        rend = renderer(jointed_mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        # convert to uint8
        rend = rend * 255
        rends.append(rend.astype("uint8"))


    imageio.mimsave(output_path, rends, fps=24, loop=0)

