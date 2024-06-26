# Use this file to generate image slice or videos

from os import path
from pathlib import Path
import subprocess
from tqdm import tqdm
import imageio
import numpy as np
import yaml

# load config file
config_file = Path("/home/siyuan/research/PoseFall/src/config.yaml")
if not config_file.exists():
    raise Exception("config file does not exist.")
# load config file
config = yaml.safe_load(config_file.open())
# input dir
# input_dir = config["generate_config"]["output_path"]
input_dir = "/home/siyuan/research/PoseFall/gen_results_exp0_2"
print(f"Loading poses from {input_dir}")
generate_images = False
generate_videos = False
generate_fbx = True


input_dir = Path(input_dir)
if not input_dir.exists():
    raise Exception("result_dir does not exist.")
# output dir
output_dir = input_dir / "blender_outputs"
all_input_files = sorted(list(input_dir.glob("*.csv")))
if generate_images:
    # list all the files in the input dir
    for csv_file in tqdm(
        all_input_files, desc="Processing files using Blender. Generate Images"
    ):
        # run blender
        subprocess.run(
            [
                "blender",
                "-b",
                "-P",
                "blender_visualize.py",
                "--",
                str(csv_file),
                str(output_dir),
                "image",
            ]
        )
        # get all generated images
        image_files = sorted(list(output_dir.glob("frame_*.png")))
        length = len(image_files)
        # select equally spaced 15 images
        step = length // 15
        image_files = image_files[: int(length * 0.8)]
        selected_images = image_files[::step]
        # read selected images
        images = [imageio.imread(str(image_file)) for image_file in selected_images]
        # crop the 10% left and right margin of the image
        margin = 0.2
        images = [image[:, int(image.shape[1] * margin) :, :] for image in images]
        # concatenate images
        image = np.concatenate(images, axis=1)
        # save the image
        imageio.imwrite(str(output_dir / f"{csv_file.stem}.png"), image)
        # delete all the images
        for image_file in image_files:
            image_file.unlink()
    print(f"All images are saved to {output_dir}")

if generate_videos:
    for csv_file in tqdm(
        all_input_files, desc="Processing files using Blender. Generate videos"
    ):
        # run blender
        subprocess.run(
            [
                "blender",
                "-b",
                "-P",
                "render_movements.py",
                "--",
                str(csv_file),
                str(output_dir),
                "video",
            ]
        )

if generate_fbx:
    for csv_file in tqdm(
        all_input_files, desc="Processing files using Blender. Generate fbx"
    ):
        # run blender
        subprocess.run(
            [
                "blender",
                "-b",
                "-P",
                "render_movements.py",
                "--",
                str(csv_file),
                str(output_dir),
                "fbx",
            ]
        )
