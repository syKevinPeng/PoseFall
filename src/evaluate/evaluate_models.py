import torch
import torch.nn as nn
import argparse, yaml
from pathlib import Path
from src.dataloader import FallingDataset1Phase
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# adapted from action2motion to take inputs of different lengths
class MotionDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, device, output_size=12, use_noise=None):
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
        motion_sequence = motion_sequence.reshape(bs, njoints*nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)


class MotionDiscriminatorForFID(MotionDiscriminator):
    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints*nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        return lin1
    

def get_model_and_dataset(args):
    dataset = FallingDataset1Phase(
    args["data_config"]["data_path"], data_aug=True, max_frame_dict=args["constant"]["max_frame_dict"]
    )
    data_configs = {"combined": dataset[0]["combined_label"].size(0)}
    print(f"Number of classes: {data_configs}")
    num_frames, num_joints, feat_dim = dataset[0]["combined_combined_poses"].size()
    print(f"Input size: {num_frames, num_joints, feat_dim}")
    data_configs.update(
        {"num_frames": num_frames, "num_joints": num_joints, "feat_dim": feat_dim}
    )
    print(f'Output size: {data_configs["combined"]}')
    model = MotionDiscriminator(data_configs["feat_dim"], hidden_size=256, hidden_layer=2, device=DEVICE, output_size=data_configs["combined"]).to(DEVICE)
    return model, dataset

def train_evaluation_model(args):
    """
    train a simple GRU model to evaluate the performance of the model
    """

    model, dataset = get_model_and_dataset(args)

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default=Path(__file__).parent.joinpath("config.yaml"),
        help="path to config file",
    )
    cmd_args = parser.parse_args()
    # load config file
    with open(cmd_args.config_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args

if __name__ == "__main__":
    args = parse_args()
    train_evaluation_model(args)