import torch, argparse, yaml
from pathlib import Path
import numpy as np

from src.evaluate import recognition_models
from .metric import calculate_accuracy, calculate_diversity_multimodality, calculate_fid
from .stgcn import STGCN
from .evaluate_dataloader import EvaluateDataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation:
    def __init__(self, num_class, recognition_model_ckpt, device, seed=None):
        model = STGCN(in_channels=6,
                      num_class=num_class,
                      graph_args={"layout": "smpl", "strategy": "spatial"},
                      edge_importance_weighting=True,
                      device=DEVICE)

        model = model.to(device)

        modelpath = Path(recognition_model_ckpt)
        if not modelpath.is_file():
            raise ValueError(f"{modelpath} is not a file.")
        
        state_dict = torch.load(modelpath, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        self.num_classes = num_class
        self.model = model
        self.device = device

        self.seed = seed

    def compute_features(self, motionloader):
        # calculate_activations_labels function from action2motion
        activations = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(motionloader):
                activations.append(self.model(batch)["features"])
                labels.append(batch["y"])
            activations = torch.cat(activations, dim=0)
            labels = torch.cat(labels, dim=0)
        return activations, labels

    @staticmethod
    def calculate_activation_statistics(activations):
        activations = activations.cpu().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate(self, dataloader):

        computedfeats = {}
        metrics = {}
        humming_score_list= []
        total_label_item = 0
        with torch.no_grad():
            for batch in dataloader:
                x = batch["combined_poses"].permute(0, 2, 3, 1)[:, :24, :, :].to(DEVICE)
                label = batch["label"].to(DEVICE)
                input_dict = {
                    "x": x,
                    "y": label,
                    "attribute_size": self.num_classes
                }
                pred = self.model(input_dict)["yhat"]
                pred = torch.sigmoid(pred).round()
                humming_score = torch.sum(pred == label).item()
                humming_score_list.append(humming_score)
                total_label_item += label.size(0)*label.size(1)
        
        metrics["humming_score"] = sum(humming_score_list)/total_label_item

        # features for diversity
        feats, labels = self.compute_features(self.model, loader)
        stats = self.calculate_activation_statistics(feats)

        computedfeats[key] = {"feats": feats,
                                "labels": labels,
                                "stats": stats}
        ret = calculate_diversity_multimodality(feats, labels, self.num_classes,
                                                seed=self.seed)
        metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret

        # taking the stats of the ground truth and remove it from the computed feats
        gtstats = computedfeats["gt"]["stats"]
        # computing fid
        for key, loader in computedfeats.items():
            metric = "fid"
            mkey = f"{metric}_{key}"

            stats = computedfeats[key]["stats"]
            metrics[mkey] = float(calculate_fid(gtstats, stats))

        return metrics
    
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

if __name__ == "__main__":
    args = parse_args()
    max_frame_dict = args["constant"]["max_frame_dict"]
    dataset = EvaluateDataset(args, args["evaluate_config"]["evaluate_dataset_path"], max_frame_dict=max_frame_dict)
    print(f'Evaluating on dataset with {len(dataset)} samples.')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    num_class = len(dataset[0]["label"])
    recognition_models_ckpt = args["evaluate_config"]["recognition_model_ckpt_path"]
    evaluation = Evaluation(num_class=num_class, recognition_model_ckpt= recognition_models_ckpt, device=DEVICE)
    metrics = evaluation.evaluate(dataloader)