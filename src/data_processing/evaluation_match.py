# match the evaluation hash 
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    # path to the csv file that contains the evaluation hash
    path_to_original_file = "./evaluation.csv"
    path_to_original_file = Path(path_to_original_file)
    assert path_to_original_file.exists(), "The file does not exist"

    # evaluated file
    path_to_evaluated_folder = "./evaluated_scores"
    path_to_evaluated_folder = Path(path_to_evaluated_folder)
    assert path_to_evaluated_folder.exists(), "The folder does not exist"
    all_evaluated_files = list(path_to_evaluated_folder.glob("*.csv"))
    evaluated_list = []
    for file in all_evaluated_files:
        evaluated_list.append(pd.read_csv(file))
    evaluated_file = pd.concat(evaluated_list, ignore_index=True)

    original_file = pd.read_csv(path_to_original_file)

    # print("Original", original_file.head())
    # print("Evaluated", evaluated_file.head())
    # merge the two files
    merged_file = original_file.merge(evaluated_file, on="hash", how="left")
    # drop the rows that don't have evaluation score
    merged_file = merged_file.dropna(subset=["smoothness score", "accuracy score", "plausibility score"])
    # get the unique model names
    model_names = merged_file["model_name"].unique()
    for model_name in model_names:
        model_df = merged_file[merged_file["model_name"] == model_name]
        avg_smoothness = model_df["smoothness score"].mean()
        avg_accuracy = model_df["accuracy score"].mean()
        avg_plausibility = model_df["plausibility score"].mean()    
        print(f'For model {model_name}, \nthe average smoothness score is {avg_smoothness},\nthe average accuracy score is {avg_accuracy},\nthe average plausibility score is {avg_plausibility}')