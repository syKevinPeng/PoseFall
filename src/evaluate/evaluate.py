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
        model = STGCN(
            in_channels=6,
            num_class=num_class,
            graph_args={"layout": "smpl", "strategy": "spatial"},
            edge_importance_weighting=True,
            device=DEVICE,
        )

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

    def evaluate(self, eval_dataloader, GT_dataloader, label_seg):
        label_seg = np.array(label_seg)[:, 1].astype(int)
        if sum(label_seg) != self.num_classes:
            raise ValueError(
                f"Number of classes in label_seg ({sum(label_seg)}) does not match num_classes ({self.num_classes})."
            )

        metrics = {}
        humming_score_list = []
        total_label_item = 0

        eval_activations = []
        gt_activations = []
        eval_labels_list = []

        with torch.no_grad():
            for eval_batch, gt_batch in zip(eval_dataloader, GT_dataloader):
                eval_x = (
                    eval_batch["combined_poses"]
                    .permute(0, 2, 3, 1)[:, :24, :, :]
                    .to(DEVICE)
                )
                eval_label = eval_batch["label"].to(DEVICE)
                eval_input_dict = {
                    "x": eval_x,
                    "y": eval_label,
                    "attribute_size": self.num_classes,
                }
                print(f'attribute size: {eval_input_dict["attribute_size"]}')
                gt_x = (
                    gt_batch["combined_combined_poses"]
                    .permute(0, 2, 3, 1)[:, :24, :, :]
                    .to(DEVICE)
                )
                gt_label = gt_batch["combined_label"].numpy().astype(int)
                gt_input_dict = {
                    "x": gt_x,
                    "y": gt_label,
                    "attribute_size": self.num_classes,
                }
                # sanity check
                if not np.array_equal(eval_label.cpu().numpy(), gt_label):
                    print(f"Eval label: {eval_label}")
                    print(f"GT label: {gt_label}")
                    raise ValueError(
                        f"Labels for eval batch and GT batch are not equal." # they have to be equal for calculating the FID score
                    )

                eval_output = self.model(eval_input_dict)
                eval_pred = eval_output["yhat"]
                eval_binarized_pred = torch.sigmoid(eval_pred).round()
                # calculate humming score
                humming_score = torch.sum(eval_binarized_pred == eval_label).item()
                # calculate per-attribute accuracy
                # save features for calculating fid
                eval_features = eval_output["features"]
                eval_activations.append(eval_features)
                eval_labels_list.append(eval_label)
                humming_score_list.append(humming_score)
                total_label_item += eval_label.size(0) * eval_label.size(1)

                # gt
                gt_output = self.model(gt_input_dict)
                gt_features = gt_output["features"]
                gt_activations.append(gt_features)

        metrics["humming_score"] = sum(humming_score_list) / total_label_item
        print(f'Humming score: {metrics["humming_score"]}')
        eval_activations = torch.cat(eval_activations, dim=0)
        eval_labels_list = torch.cat(eval_labels_list, dim=0)

        # features for diversity
        eval_stats = self.calculate_activation_statistics(eval_activations)
        eval_labels_list = eval_labels_list.cpu().numpy()
        diversity = calculate_diversity(
            eval_activations, eval_labels_list, self.num_classes, seed=self.seed
        )
        print(f"Diversity: {diversity}")
        metrics["diversity"] = diversity

        gt_activations = torch.cat(gt_activations, dim=0)
        gtstats = self.calculate_activation_statistics(gt_activations)
        fid = calculate_fid(gtstats, eval_stats)
        print(f"FID: {fid}")
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
            args=args,
            data_path=args["data_config"]["data_path"],
            data_aug=False,
            max_frame_dict=args["constant"]["max_frame_dict"],
            split="all",
        )

    def __len__(self):
        return len(self.dataset) * self.num_repeats

    # since generated images are repeated num_repeats times, we need to extend the GT dataset to match the generated dataset
    def __getitem__(self, idx):
        return self.dataset[idx // self.num_repeats]


if __name__ == "__main__":
    args = parse_args()
    max_frame_dict = args["constant"]["max_frame_dict"]
    eval_dataset = EvaluateDataset(
        args,
        args["evaluate_config"]["evaluate_dataset_path"],
        max_frame_dict=max_frame_dict,
    )
    label_seg = eval_dataset.label_seg
    print(f"Evaluating on dataset with {len(eval_dataset)} samples.")
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    gt_dataset = EvalGTDataset(args)
    gt_dataloader = torch.utils.data.DataLoader(
        gt_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    num_class = len(eval_dataset[0]["label"])
    recognition_models_ckpt = args["evaluate_config"]["recognition_model_ckpt_path"]
    evaluation = Evaluation(
        num_class=num_class,
        recognition_model_ckpt=recognition_models_ckpt,
        device=DEVICE,
    )
    metrics = evaluation.evaluate(eval_dataloader, gt_dataloader, label_seg)
