# %%
from asyncore import loop
import numpy as np
from pathlib import Path
from icecream import ic
# # %%
# DATAPATH = '../data/falling_dataset.npz'
VIZ_OUTPUT = '/home/siyuan/research/PoseFall/src/visulization/viz_output'
# DATAPATH = Path(DATAPATH)
VIZ_OUTPUT = Path(VIZ_OUTPUT)
if not VIZ_OUTPUT.is_dir():
    VIZ_OUTPUT.mkdir()


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
    rend = renderer(viz_mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # convert to uint8
    rend = rend * 255
    rends.append(rend.astype("uint8"))


imageio.mimsave(VIZ_OUTPUT/"example.gif", rends, fps=24, loop=0)


# %%
joints.shape