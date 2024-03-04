"""
This file is used to postprocess the output of the model to make it ready for evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import tqdm


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

def upload_file(file_path, hash, folder_id):
    file = drive.CreateFile(
        {
            "title": hash,
            "parents": [{"kind": "drive#fileLink", "id": folder_id}],
        }
    )
    file.SetContentFile(file_path)
    file.Upload()
    file.InsertPermission({"type": "anyone", "value": "anyone", "role": "reader"})
    shareable_link = file["alternateLink"]
    return [shareable_link, hash]


if __name__ == "__main__":
    config_file_path = Path("/home/siyuan/research/PoseFall/src/config.yaml")
    config = yaml.safe_load(config_file_path.read_text())

    concate_3E3DRNN_path = Path(
        "/home/siyuan/research/PoseFall/gen_results_exp13/blender_outputs/video_rendering"
    )
    addition_3E3DRNN_path = Path(
        "/home/siyuan/research/PoseFall/gen_results_exp14/blender_outputs/video_rendering"
    )
    ACTOR_path = Path(
        "/home/siyuan/research/PoseFall/gen_results_exp0/blender_outputs/video_rendering"
    )
    path_list = [concate_3E3DRNN_path, addition_3E3DRNN_path, ACTOR_path]

    if not concate_3E3DRNN_path.is_dir():
        raise FileNotFoundError(f"{concate_3E3DRNN_path} is not a valid directory.")
    if not ACTOR_path.is_dir():
        raise FileNotFoundError(f"{ACTOR_path} is not a valid directory.")

    upload = False

    if upload:
        data_1 = iter_through_path("concate_3E3DRNN", concate_3E3DRNN_path, config)
        data_2 = iter_through_path("ACTOR", ACTOR_path, config)
        data_3 = iter_through_path("addition_3E3DRNN", addition_3E3DRNN_path, config)

        data = data_1 + data_2 + data_3
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

        # upload to google drive
        from pydrive.drive import GoogleDrive
        from pydrive.auth import GoogleAuth

        # test file upload
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        # destination folder id
        folder_id = "1bCww8Tcna2bCGuh4o6VRbMq698P60szv"
        shareable_links = []
        import concurrent.futures
        shareable_links = []
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for row in df.iterrows():
                file_path = row[1]["file_path"]
                hash = row[1]["hash"]
                future = executor.submit(upload_file, file_path, hash, folder_id)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                shareable_link = future.result()
                shareable_links.append(shareable_link)
        
        df["link"] = [link[0] for link in shareable_links]
        df["upload_hash"] =[link[1] for link in shareable_links]
        df["evaluation score"] = None
        # cleanning the csv to match upload hash with the hash
        df = pd.read_csv("evaluation.csv")
        df_1 = df.drop(columns=["link", "upload_hash"])
        df_2 = df[['link', 'upload_hash']]
        # merge df 1 and df2 on hash and upload_hash
        cleaned_df = pd.merge(df_1, df_2, left_on="hash", right_on="upload_hash")
        print(cleaned_df.head())
        cleaned_df = cleaned_df.drop(columns=["upload_hash"])
        cleaned_df.to_csv("evaluation.csv", index=False)
        print("The csv file is cleaned and saved to evaluation.csv")

    else:
        df = pd.read_csv("evaluation.csv")
        # shuffle the dataframe
        df = df.sample(frac=1, random_state=42)
        # drop model_name and file_path
        df = df.drop(columns=["model_name", "file_path"])
        # move the evaluation score to the last column
        cols = list(df.columns)
        cols.remove("evaluation score")
        cols.append("evaluation score")
        df = df[cols]
        # evenly split the dataframe into 7 
        n = 7
        df_list = np.array_split(df, n)
        # save the dataframes to csv
        for i, df in enumerate(df_list):
            df.to_csv(f"./evaluation_splits/evaluation_{i}.csv", index=False)


        

    
