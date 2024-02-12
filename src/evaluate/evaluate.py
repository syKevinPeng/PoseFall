import torch, argparse, yaml
from pathlib import Path
import numpy as np

from src.evaluate import recognition_models
from .metric import calculate_accuracy, calculate_diversity, calculate_fid
from .stgcn import STGCN
from .evaluate_dataloader import EvaluateDataset
from ..dataloader import FallingDataset1Phase
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

    def evaluate(self, eval_dataloader, GT_dataloader):

        computedfeats = {}
        metrics = {}
        humming_score_list= []
        total_label_item = 0

        activations = []
        labels = []
        with torch.no_grad():
            for batch in eval_dataloader:
                x = batch["combined_poses"].permute(0, 2, 3, 1)[:, :24, :, :].to(DEVICE)
                label = batch["label"].to(DEVICE)
                input_dict = {
                    "x": x,
                    "y": label,
                    "attribute_size": self.num_classes
                }
                output = self.model(input_dict)
                pred = output["yhat"]
                binarized_pred = torch.sigmoid(pred).round()
                humming_score = torch.sum(binarized_pred == label).item()
                features = output["features"]
                activations.append(features)
                labels.append(label)
                humming_score_list.append(humming_score)
                total_label_item += label.size(0)*label.size(1)
        
        metrics["humming_score"] = sum(humming_score_list)/total_label_item
        print(f'Humming score: {metrics["humming_score"]}')
        activations = torch.cat(activations, dim=0)
        labels = torch.cat(labels, dim=0)

        # features for diversity
        stats = self.calculate_activation_statistics(activations)
        labels = labels.cpu().numpy()
        diversity = calculate_diversity(activations, labels, self.num_classes,
                                                 seed=self.seed)
        print(f'Diversity: {diversity}')
        metrics["diversity"] = diversity

        if len(eval_dataloader) != len(GT_dataloader):
            raise ValueError(f'Length of eval dataloader {len(eval_dataloader)} and GT dataloader {len(GT_dataloader)} are not equal.')
        for i, (eval_batch, GT_batch) in enumerate(zip(eval_dataloader, GT_dataloader)):
            eval_label = eval_batch["label"].cpu().numpy()
            GT_label = GT_batch["combined_label"].cpu().numpy().astype(int)
            if not np.array_equal(eval_label, GT_label):
                print(f'Eval label: {eval_label}')
                print(f'GT label: {GT_label}')
                raise ValueError(f'Labels for eval batch {i} and GT batch {i} are not equal.')
        gt_activations = []
        #run stats on GT data
        with torch.no_grad():
            for batch in GT_dataloader:
                x = batch["combined_combined_poses"].permute(0, 2, 3, 1)[:, :24, :, :].to(DEVICE)
                label = batch["combined_label"].to(DEVICE)
                input_dict = {
                    "x": x,
                    "y": label,
                    "attribute_size": self.num_classes
                }
                output = self.model(input_dict)
                features = output["features"]
                gt_activations.append(features)

        gt_activations = torch.cat(gt_activations, dim=0)
        gtstats = self.calculate_activation_statistics(gt_activations)
        fid = calculate_fid(gtstats, stats)
        print(f'FID: {fid}')
        metrics["fid"] = fid

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

class EvalGTDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.num_repeats = args["generate_config"]["num_to_gen"]
        self.dataset = FallingDataset1Phase(
        args["data_config"]["data_path"], data_aug=False, max_frame_dict=args["constant"]["max_frame_dict"], split="all")
        


    def __len__(self):
        return len(self.dataset)*self.num_repeats

    # since generated images are repeated num_repeats times, we need to extend the GT dataset to match the generated dataset
    def __getitem__(self, idx):
        return self.dataset[idx // self.num_repeats]

if __name__ == "__main__":
    args = parse_args()
    max_frame_dict = args["constant"]["max_frame_dict"]
    eval_dataset = EvaluateDataset(args, args["evaluate_config"]["evaluate_dataset_path"], max_frame_dict=max_frame_dict)
    print(f'Evaluating on dataset with {len(eval_dataset)} samples.')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)
    gt_dataset = EvalGTDataset(args)
    gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=32, shuffle=False, num_workers=4)
    num_class = len(eval_dataset[0]["label"])
    recognition_models_ckpt = args["evaluate_config"]["recognition_model_ckpt_path"]
    evaluation = Evaluation(num_class=num_class, recognition_model_ckpt= recognition_models_ckpt, device=DEVICE)
    metrics = evaluation.evaluate(eval_dataloader, gt_dataloader)