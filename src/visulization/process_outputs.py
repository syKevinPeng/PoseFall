from pathlib import Path
import subprocess
from tqdm import tqdm
import imageio
import numpy as np
# input dir
input_dir = "/home/siyuan/research/PoseFall/gen_results_3E3D_20"
generate_images = False
generate_videos = True



input_dir = Path(input_dir)
if not input_dir.exists():
    raise Exception("result_dir does not exist.")
# output dir
output_dir = input_dir/"blender_outputs"
all_input_files = sorted(list(input_dir.glob("*.csv")))
if generate_images:
# list all the files in the input dir
    for csv_file in tqdm(all_input_files, desc="Processing files using Blender. Generate Images"):
        # run blender
        subprocess.run(["blender", "-b", "-P", "blender_visualize.py", "--", str(csv_file), str(output_dir), "image"])
        # get all generated images
        image_files = sorted(list(output_dir.glob("frame_*.png")))
        length = len(image_files)
        # select equally spaced 15 images
        step = length // 10
        selected_images = image_files[::step]
        # read selected images
        images = [imageio.imread(str(image_file)) for image_file in selected_images]
        # crop the 10% left and right margin of the image
        margin = 0.3
        images = [image[:,int(image.shape[1]*margin):, :]for image in images]
        # concatenate images
        image = np.concatenate(images, axis=1)
        # save the image
        imageio.imwrite(str(output_dir/f"{csv_file.stem}.png"), image)
        # delete all the images
        for image_file in image_files:
            image_file.unlink()
    print(f'All images are saved to {output_dir}')

if generate_videos:
    for csv_file in tqdm(all_input_files, desc="Processing files using Blender. Generate videos"):
        # run blender
        subprocess.run(["blender", "-b", "-P", "blender_visualize.py", "--", str(csv_file), str(output_dir), "video"])
        exit()



