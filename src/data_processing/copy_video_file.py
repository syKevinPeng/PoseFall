# extract .avi files
import os
import shutil
from pathlib import Path


source_dir = Path("E:/Downloads/Falling_dataset_session3_181-200/Falling_dataset_session3_181-200")
destination_dir = Path("E:/Downloads/Mocap_data_video_2")
if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)
if not os.path.exists(source_dir):
    raise Exception("No source directory found")

for trial_dir in os.listdir(source_dir):
    if trial_dir.startswith("Trial_"):
        trial_number = trial_dir.split("_")[1]
        trial_path = os.path.join(source_dir, trial_dir)
        for file in os.listdir(trial_path):
            if file == "stream01.avi":
                file_name = f"Trial_{trial_number}_{file}"
                source_file = os.path.join(trial_path, file)
                destination_file = os.path.join(destination_dir, file_name)
                shutil.copyfile(source_file, destination_file)
