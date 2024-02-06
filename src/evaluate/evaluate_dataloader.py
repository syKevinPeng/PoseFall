import torch
from torch.utils.data import Dataset
from pathlib import Path
from .stgcn import STGCN
import argparse, yaml
from tqdm import tqdm
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvaluateDataset(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = Path(data_path)

        if not self.data_path.is_dir():
            raise ValueError(f"{self.data_path} is not a directory.")
        
        self.data_list = list(self.data_path.glob("*.csv"))
        num_class = len(self.data_list[0].stem.split("_")[0])
        print(f"Number of classes: {num_class}")
        self.recognition_model =  STGCN(in_channels=6, 
                  num_class=num_class, 
                  graph_args={"layout": "smpl", "strategy": "spatial"},
                  edge_importance_weighting=True, 
                  device=DEVICE).to(DEVICE)

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"x": self.data[idx], "y": self.labels[idx]}
    
def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default=Path(__file__).parent.parent.joinpath("config.yaml"),
        help="path to config file",
    )
    cmd_args = parser.parse_args()
    # load config file
    with open(cmd_args.config_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args

if __name__=="__main__":
    args = parse_args()
    # prepare wandb
    # wandb.init(
    #     project="posefall_recognition",
    #     config=args,
    #     mode=wandb_config["wandb_mode"],
    #     tags=wandb_config["wandb_tags"],
    #     name="recognition_training",
    #     notes="training recognition model with GT data",
    # )

    EvaluateDataset(args, args["evaluate_config"]["evaluate_dataset_path"])