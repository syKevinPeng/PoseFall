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
    # annotation = [str(trial_number),
    #                 data["annotations"][0]["name"], # attribute
    #                 *data["annotations"][0]["ranges"][0], # ranges
    #                 data["annotations"][1]["name"], # attribute 
    #                 *data["annotations"][1]["ranges"][0], # ranges
    #                 data["annotations"][2]["name"], # attribute
    #                 *data["annotations"][2]["ranges"][0]] # ranges
    annotation = {trial_number:{
                                        data["annotations"][0]["name"]:data["annotations"][0]["ranges"][0],
                                        data["annotations"][1]["name"]:data["annotations"][1]["ranges"][0],
                                        data["annotations"][2]["name"]:data["annotations"][2]["ranges"][0]
                                            } 
                                        }
    annotations_dicts.update(annotation)

# convert to csv
annotations = [[key,*annotations_dicts[key]["Impact"], *annotations_dicts[key]["Glitch"], *annotations_dicts[key]["Fall"] ] for key in annotations_dicts.keys()]
print(annotations[0])
col_name = ["Trial Number", "Impact Start","Impact End", "Glitch Start","Glitch End","Fall Sart","Fall End"]
annotations = pd.DataFrame(annotations, columns=col_name)
print(annotations.head())
