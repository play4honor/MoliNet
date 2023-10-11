import yaml
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from src.net import AstrosNet
from src.data import PitchSequenceDataset


class TopKAccuracy:
    def __init__(self, k):
        self.k = k
        self.n_correct = torch.tensor([0.0], device=net.device)
        self.n_total = torch.tensor([0.0], device=net.device)

    def update(self, preds, targets):
        _, topk_idx = torch.topk(preds, k=self.k, dim=-1)
        top_5_accurate = torch.eq(
            topk_idx,
            targets.unsqueeze(-1),
        ).any(dim=-1)
        self.n_correct += top_5_accurate.sum()
        self.n_total += top_5_accurate.numel()

    def __call__(self):
        return (self.n_correct / self.n_total).item()

    def __str__(self):
        return str((self.n_correct / self.n_total).item())


print("Reading config")
# This is relative to the root directory, not astros/
with open("./astros/astros_config.yaml", "r") as f:
    config = yaml.load(f, yaml.CLoader)

# We also need to load the morpher state dicts.
with open("./reference/morphers.yaml", "r") as f:
    morpher_states = yaml.load(f, yaml.CLoader)

# Read data
pitch_df = pl.read_parquet("./data/bigger_training_data.parquet")
# Filter games with mostly missing pitch data
pitch_df = pitch_df.filter(pl.col("pitch_type").is_null().sum().over("game_pk") <= 30)

# Create the dataset. We'll use the loaded morpher states.
ds = PitchSequenceDataset(
    pitch_df,
    feature_config=config["features"],
    sequence_length=config["net_params"]["sequence_length"],
    morpher_states=morpher_states,
)

# #tooHighAPrice
disjoint_indices = (a := torch.arange(len(ds)))[
    a % config["net_params"]["sequence_length"] == 0
]
ds = Subset(ds, disjoint_indices)
dl = DataLoader(ds, batch_size=256)

net = AstrosNet.load_from_checkpoint(
    "./astros/checkpoint/epoch=6-step=15547.ckpt",
    **config["net_params"],
    optim_lr=config["training_params"]["learning_rate"],
    morphers=ds.dataset.morphers,
)
net.eval()

accuracies = {
    1: TopKAccuracy(1),
    5: TopKAccuracy(5),
    10: TopKAccuracy(10),
}

new_accuracies = {
    1: TopKAccuracy(1),
    5: TopKAccuracy(5),
    10: TopKAccuracy(10),
}

first_in_sequence_accuracies = {
    1: TopKAccuracy(1),
    5: TopKAccuracy(5),
    10: TopKAccuracy(10),
}

print("Scoring Dataset")

for i, (x, is_sog) in enumerate(dl):
    ground_truth_pitchers = x["pitcher"][:, 1:].to(net.device)
    preds = net({k: v.to(net.device) for k, v in x.items()}, is_sog.to(net.device))
    pitcher_preds = F.softmax(preds["pitcher"][:, :-1, :], dim=-1)

    # Get the pitchers where a pitcher appears for the first time in the sequence
    # n x s x 1
    gt_1 = x["pitcher"].to(net.device).unsqueeze(dim=-1)
    # n x 1 x s
    gt_2 = x["pitcher"].to(net.device).unsqueeze(dim=1)
    # n x s x s
    pitcher_is_equal = torch.eq(gt_1, gt_2)
    # Mask the upper triangle and the main diagonal
    masked_pitcher_is_equal = torch.tril(pitcher_is_equal, diagonal=-1)
    # n x s
    is_first_in_sequence = masked_pitcher_is_equal.any(dim=1)[:, 1:]

    for a in accuracies.values():
        a.update(pitcher_preds, ground_truth_pitchers)

    # For new pitcher (compared to last pitch, not in game)
    is_new_pitcher = ~torch.eq(
        ground_truth_pitchers, x["pitcher"][:, :-1].to(net.device)
    )
    for a in new_accuracies.values():
        a.update(pitcher_preds[is_new_pitcher], ground_truth_pitchers[is_new_pitcher])

    # For new pitcher (in sequence)
    # Get the pitchers where a pitcher appears for the first time in the sequence
    # n x s x 1
    gt_1 = ground_truth_pitchers.unsqueeze(dim=-1)
    # n x 1 x s
    gt_2 = ground_truth_pitchers.unsqueeze(dim=1)
    # n x s x s
    pitcher_is_equal = torch.eq(gt_1, gt_2)
    # Mask the upper triangle and the main diagonal
    masked_pitcher_is_equal = torch.tril(pitcher_is_equal, diagonal=-1)
    # n x s
    is_first_in_sequence = ~masked_pitcher_is_equal.any(dim=-1)

    for a in first_in_sequence_accuracies.values():
        a.update(
            pitcher_preds[is_first_in_sequence],
            ground_truth_pitchers[is_first_in_sequence],
        )


for k, a in accuracies.items():
    print(f"Top {k} accuracy: {a() : .3f}")

for k, a in new_accuracies.items():
    print(f"Top {k} (new pitcher) accuracy: {a() : .3f}")

for k, a in first_in_sequence_accuracies.items():
    print(f"Top {k} (new in sequence pitcher) accuracy: {a() : .3f}")
