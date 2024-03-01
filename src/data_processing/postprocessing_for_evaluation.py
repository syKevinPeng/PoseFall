"""
This file is used to postprocess the output of the model to make it ready for evaluation.
"""

from pathlib import Path    
import numpy as np
import pandas as pd
import yaml

def read_label_from_file_name(file_name: str, label_str):
    """
    This function reads the label from the file name.
    """
    label_onehot = str(file_name).split("_")[-3]
    # check label str only contains zeros and ones
    assert all([char in ["0", "1"] for char in label_onehot])
    # get the index of 1
    label_index = [i for i, char in enumerate(label_onehot) if char == "1"]
    # get the label
    label = [label_str[i] for i in label_index]
    return label

def get_label_list(config_file):
    mylist = []
    data_path = Path(config_file["data_config"]["data_path"])
    label_path = data_path / "label.csv"
    label_df = pd.read_csv(label_path, header=0)
    all_labels = label_df.columns[1:]
    # load selected attributes
    impact_att = config_file["constant"]["attributes"]["impact_phase_att"]
    glitch_att = config_file["constant"]["attributes"]["glitch_phase_att"]
    fall_att = config_file["constant"]["attributes"]["fall_phase_att"]
    for attri in np.concatenate([impact_att, glitch_att, fall_att]):
        # count how many label in all_labels contains the string in attri
        for label in all_labels:
            if attri in label:
                mylist.append(label)
    return mylist
    
    
def iter_through_path(model_name: str, path: Path, config: dict):
    """
    This function iterates through the path and returns the data in the path.
    """
    data = []
    label_str = get_label_list(config)
    for file in path.iterdir():
        label = read_label_from_file_name(file, label_str)
        data.append([model_name, label, str(file)])
    return data

if __name__ == "__main__":
    config_file_path = Path("/home/siyuan/research/PoseFall/src/config.yaml")
    config = yaml.safe_load(config_file_path.read_text())

    concate_3E3DRNN_path = Path("/home/siyuan/research/PoseFall/gen_results_exp13/blender_outputs/video_rendering")
    addition_3E3DRNN_path = None
    ACTOR_path = Path("/home/siyuan/research/PoseFall/gen_results_exp0/blender_outputs/video_rendering")
    path_list = [concate_3E3DRNN_path, ACTOR_path]

    if not concate_3E3DRNN_path.is_dir():
        raise FileNotFoundError(f"{concate_3E3DRNN_path} is not a valid directory.")
    if not ACTOR_path.is_dir():
        raise FileNotFoundError(f"{ACTOR_path} is not a valid directory.")
    
    data_1= iter_through_path("concate_3E3DRNN", concate_3E3DRNN_path, config)
    data_2= iter_through_path("ACTOR", ACTOR_path, config)
    data = data_1 + data_2
    df = pd.DataFrame(data, columns=["model_name", "label", "file_path"])
    # split the label into multiple columns
    df = pd.concat([df, df["label"].apply(pd.Series)], axis=1)
    # drop the label column
    df = df.drop(columns=["label"])
    # give name to the columns
    for col in df.columns[2:]:
        col_name = df.iloc[0][col].split("_")[0]
        df = df.rename(columns={col: col_name})
        df[col_name] = df[col_name].str.split("_").str[1]
    # create a hash column
    df["hash"] = pd.util.hash_pandas_object(df, index=False)
    # print(df.head())

    # upload to google drive
    from pydrive.drive import GoogleDrive 
    from pydrive.auth import GoogleAuth 
    # test file upload
    gauth = GoogleAuth() 
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # destination folder id
    folder_id = "1TSwAL4mRYKunZeDb8mBMeWcHvbFSpscp" 
    test_file_path = "/home/siyuan/research/PoseFall/gen_results_exp14/blender_outputs/video_rendering/0000_00100010000001000000100_sequences_0.mp4"
    test_file = drive.CreateFile({
        "title": "test.mp4",
        "parents": [{"kind": "drive#fileLink", "id": folder_id}]
    })
    test_file.SetContentFile(test_file_path)
    test_file.Upload()
    print(f'Uploaded {test_file_path} to Google Drive.')

    


