from pathlib import Path
import subprocess
from tqdm import tqdm
# input dir
result_dir = "/home/siyuan/research/PoseFall/src/gen_results3E3D"
result_dir = Path(result_dir)
if not result_dir.exists():
    raise Exception("result_dir does not exist.")
# output dir
output_dir = result_dir/"blender_outputs"

# list all the files in the input dir
result_files = sorted(list(result_dir.glob("*.csv")))
for csv_file in tqdm(result_files, desc="Processing files using Blender"):
    # run blender
    subprocess.run(["blender", "-b", "-P", "blender_visualize.py", "--", str(csv_file), str(output_dir)])
    break

