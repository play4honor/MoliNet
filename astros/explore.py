from src.net import AstrosNet
import yaml
import polars as pl
from src.data import PitchSequenceDataset
import torch
import torch.nn.functional as F

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

net = AstrosNet.load_from_checkpoint(
    "./lightning_logs/version_21/checkpoints/epoch=0-step=2221.ckpt",
    **config["net_params"],
    optim_lr=config["training_params"]["learning_rate"],
    morphers=ds.morphers,
    strict=False
)

x, is_sog = ds[150]

preds = net({k: v.to(net.device) for k, v in x.items()}, is_sog.to(net.device))

def fixed_ce_loss(input, target):
    return F.cross_entropy(
        input.permute(0, 2, 1), target, reduction="none"
    )

#print(fixed_ce_loss(preds["pitcher"], x["pitcher"].to(net.device).unsqueeze(0)).mean())

max_vals, max_idx = torch.max(F.softmax(preds["pitcher"], dim=-1), dim=-1)

for v, i, a in zip(max_vals.squeeze(), max_idx.squeeze(), x["pitcher"].squeeze()):
    print(f"{v :.3f}, {i}, {a}")
