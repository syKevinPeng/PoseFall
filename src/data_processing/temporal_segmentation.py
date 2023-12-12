"""
This script takes annotation format: Darwin 2.0 JSON (https://docs.v7labs.com/reference/darwin-json) 
and convert it to csv file
"""

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

input_dir = Path("/home/siyuan/research/PoseFall/data/MoCap/Mocap_data/temporal_annotation")
output_dir = Path("/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data")

if not input_dir.exists():
    raise ValueError(f"Input directory {input_dir} does not exist.")
if not output_dir.exists():
    raise ValueError(f"Output directory {output_dir} does not exist.")


# get all json files in the directory
json_list = [f for f in input_dir.glob('*.json')]
json_list.sort()

annotations_dicts={}
# loop through all the json files
for file_path in json_list:
    with open(file_path) as f:
        data = json.load(f)
    # get the trial number
    trial_number = re.search(r'Trial_(\d+)', file_path.stem).group(1)
    print(f"Start to process Trial {trial_number}")
    annotation = {trial_number:{
                                        data["annotations"][0]["name"]:data["annotations"][0]["ranges"][0],
                                        data["annotations"][1]["name"]:data["annotations"][1]["ranges"][0],
                                        data["annotations"][2]["name"]:data["annotations"][2]["ranges"][0]
                                            } 
                                        }
    annotations_dicts.update(annotation)

# convert to csv
annotations = [[key,*annotations_dicts[key]["Impact"], *annotations_dicts[key]["Glitch"], *annotations_dicts[key]["Fall"] ] for key in annotations_dicts.keys()]
col_name = ["Trial Number", "Impact Start","Impact End", "Glitch Start","Glitch End","Fall Sart","Fall End"]
annotations_df = pd.DataFrame(annotations, columns=col_name)
# do some post processing. 
# find the maximum gap
gap_imp_gli = annotations_df["Glitch Start"] - annotations_df["Impact End"]
gap_gli_fall = annotations_df["Fall Sart"] - annotations_df["Glitch End"]
print(f'max gap between impact and glitch is {gap_imp_gli.max()}')
print(f'max gap between glitch and fall is {gap_gli_fall.max()}')
# set the glitch start = impact end + 1
annotations_df["Glitch Start"] = annotations_df["Impact End"] + 1
# set the fall start = glitch end + 1
annotations_df["Fall Sart"] = annotations_df["Glitch End"] + 1
gap_imp_gli = annotations_df["Glitch Start"] - annotations_df["Impact End"]
gap_gli_fall = annotations_df["Fall Sart"] - annotations_df["Glitch End"]
# append phase information to the csv
# read all preprocessed data
preprocessed_data = sorted([f for f in (output_dir/"preprocessed_data").glob('Trial_*.csv')])
for anno in annotations:
    print(f'adding temporal information to trial {anno[0]}')
    # read the preprocessed data
    data = pd.read_csv(output_dir/"preprocessed_data"/f"Trial_{anno[0]}.csv")
    # construct a new columns for phase information
    phase_col = np.full(data.shape[0], "none")
    # impact phase
    phase_col[anno[1]:anno[2]+1] = "impact"
    # glitch phase
    phase_col[anno[3]:anno[4]+1] = "glitch"
    # fall phase   
    phase_col[anno[5]:anno[6]+1] = "fall"
    data["phase"] = phase_col
    # save the new csv file
    data.to_csv(output_dir/f"Trial_{anno[0]}.csv", index=False)