# %%
from asyncore import loop
import numpy as np
from pathlib import Path
from icecream import ic
from scipy.datasets import face
# %%
DATAPATH = '../data/falling_dataset.npz'
VIZ_OUTPUT = 'viz_output'
DATAPATH = Path(DATAPATH)
VIZ_OUTPUT = Path(VIZ_OUTPUT)
if DATAPATH.is_file():
    dataset = np.load(DATAPATH, allow_pickle=True)['falling_dataset'][()]
else:
    raise FileNotFoundError('Data file not found.')

# if not VIZ_OUTPUT.is_dir():
#     VIZ_OUTPUT.mkdir()
# # %%
# # second trial:
# trial = dataset[2]
# trial.shape
# # %%
# frame = trial[0]

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure( figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=0, azim=-90)
# joints = frame
# for i in range(joints.shape[0]):
#     ax.scatter(joints[i, 0], joints[i, 1], joints[i, 2], c='r', marker='o')
#     ax.text(joints[i, 0] + 0.02, joints[i, 1], joints[i, 2], str(i+1), fontsize = 10)

# # set x scale
# ax.set_xlim3d(-0.5, 0.5)
# # set y scale
# ax.set_ylim3d(-1, 1)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()



# %%
'''
SMPL Config
0: 'pelvis',
1: 'left_hip',
2: 'right_hip',
3: 'spine1',
4: 'left_knee',
5: 'right_knee',
6: 'spine2',
7: 'left_ankle',
8: 'right_ankle',
9: 'spine3',
10: 'left_foot',
11: 'right_foot',
12: 'neck',
13: 'left_collar',
14: 'right_collar',
15: 'head',
16: 'left_shoulder',
17: 'right_shoulder',
18: 'left_elbow',
19: 'right_elbow',
20: 'left_wrist',
21: 'right_wrist',
22: 'left_hand',
23: 'right_hand'
'''
# %%
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
print(mesh)
# %%
import pytorch3d, torch, imageio
# visulize use pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
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
# check if variable is on GPU
ic(viz_mesh.device)
ic(textures.device)
ic(faces.device)
ic(vertices.device)
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
