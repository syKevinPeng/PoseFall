import pathlib
import attr
from .metric import compute_accuracy, compute_hamming_score
import torch
import torch.nn as nn
import argparse, yaml
from tqdm import tqdm
from pathlib import Path
from ..dataloader import FallingDataset1Phase, FallingDataset3Phase
from .stgcn import STGCN
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# adapted from action2motion to take inputs of different lengths
class MotionDiscriminator(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_layer,
        device,
        output_size=12,
        use_noise=None,
    ):
        super(MotionDiscriminator, self).__init__()
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints * nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[
            tuple(torch.stack((lengths - 1, torch.arange(bs, device=self.device))))
        ]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2

    def initHidden(self, num_samples, layer):
        return torch.randn(
            layer,
            num_samples,
            self.hidden_size,
            device=self.device,
            requires_grad=False,
        )


class MotionDiscriminatorForFID(MotionDiscriminator):
    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints * nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[
            tuple(torch.stack((lengths - 1, torch.arange(bs, device=self.device))))
        ]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        return lin1
    

def get_model_and_dataloader(args):
    train_dataset = FallingDataset1Phase(
            args = args,
            data_path = args["data_config"]["data_path"],
            data_aug = True,
            max_frame_dict = args["constant"]["max_frame_dict"],
            split="train",
        )
    # since we are dealing with all phases
    phase = "combined"
    data_configs = {"num_class": train_dataset[0][f"{phase}_label"].size(0)}
    num_frames, num_joints, feat_dim = train_dataset[0][f"{phase}_combined_poses"].size()
    data_configs.update(
        {"num_frames": num_frames, "num_joints": num_joints, "feat_dim": feat_dim}
    )
    print(f'Output size: {data_configs["num_class"]}')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["recognition_config"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    attr_size = train_dataset.get_attr_size()
    print(f'Attribute size: {attr_size}')
    num_class = data_configs["num_class"]
    # load STGCN model
    model = STGCN(
        in_channels=feat_dim,
        num_class=num_class,
        graph_args={"layout": "smpl", "strategy": "spatial"},
        edge_importance_weighting=True,
        phase_output_size=attr_size,
        device=DEVICE,
    ).to(DEVICE)

    eval_dataset = FallingDataset1Phase(
        args = args,
        data_path = args["data_config"]["data_path"],
        data_aug = True,
        max_frame_dict = args["constant"]["max_frame_dict"],
        split="all",
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args["recognition_config"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    # model = MotionDiscriminator(data_configs["feat_dim"], hidden_size=256, hidden_layer=2, device=DEVICE, output_size=data_configs["combined"]).to(DEVICE)
    return model, train_dataloader, eval_dataloader, attr_size, num_class


def train_evaluation_model(
    args
):
    """
    train a simple GRU model to evaluate the performance of the model
    """
    output_dir = Path(args["recognition_config"]["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    recognition_config = args["recognition_config"]
    model, train_dataloader, evaluate_dataloader, attr_sizes, num_class = get_model_and_dataloader(args)

    # load pretrain weights
    state_dict = torch.load(
        recognition_config["pretrained_weights"], map_location=DEVICE
    )
    # remove unwanted keys
    kets_to_remove = ["fcn.weight", "fcn.bias"]
    for key in kets_to_remove:
        state_dict.pop(key, None)
    model.load_state_dict(state_dict, strict=False)
    # training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=recognition_config["lr"])

    for epoch in range(recognition_config["epochs"]):
        epoch_loss = 0
        cum_loss_dict = {}
        for i_batch, data_dict in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()   
            # input_shape:  Batch, Num Joints, Angle Rep (6), Time
            x = data_dict[f"combined_combined_poses"].permute(0, 2, 3, 1)[:, :24, :, :]
            input_dict = {
                "x": x.to(DEVICE),
                "y": data_dict[f"combined_label"].to(DEVICE),
                "attribute_size": attr_sizes,
            }
            batch_output = model(input_dict)
            # add a softmax layer to the output
            loss, loss_dict = model.compute_loss(batch_output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            for key in loss_dict.keys():
                if key not in cum_loss_dict.keys():
                    cum_loss_dict[key] = loss_dict[key]
                else:
                    cum_loss_dict[key] += loss_dict[key]
        print(f"Epoch {epoch} loss: {epoch_loss}. Loss dict: {loss_dict}")
        wandb.log({"epoch_loss": epoch_loss})
        for key in loss_dict.keys():
            wandb.log({f"epoch_{key}": loss_dict[key]})

        if (epoch + 1) % recognition_config["model_save_freq"] == 0:
            torch.save(model.state_dict(), output_dir / f"recognition_model_{epoch}.pt")

        if (epoch + 1) % recognition_config["evaluation_freq"] == 0:
            corr_pred_counter = 0
            total_label_item = 0
            with torch.no_grad():
                for batch in evaluate_dataloader:
                    x = batch[f"combined_combined_poses"].permute(0, 2, 3, 1)[:, :24, :, :].to(DEVICE)
                    label = batch[f"combined_label"].to(DEVICE)
                    input_dict = {"x": x, "y": label, "attribute_size": num_class}
                    batch = model(input_dict)
                    for i, attr in enumerate(attr_sizes):
                        phase_pred = batch["yhat"][i]
                        phase_gt = batch["y"][:,sum(attr_sizes[:i]): sum(attr_sizes[:i]) + attr]
                        num_correct, num = model.compute_accuracy(phase_pred, phase_gt)
                        corr_pred_counter += num_correct
                        total_label_item += num

            accuracy = corr_pred_counter / total_label_item
            print(f'Evaluate Accuracy: {accuracy}')
            wandb.log({"evaluate accuracy": accuracy})


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
    wandb_config = args["recognition_config"]["wandb_config"]
    # prepare wandb
    wandb.init(
        project="posefall_recognition",
        config=args,
        mode=wandb_config["wandb_mode"],
        tags=wandb_config["wandb_tags"],
        name="recognition_training",
        notes=wandb_config["wandb_description"],
        
    )
    train_evaluation_model(args)
