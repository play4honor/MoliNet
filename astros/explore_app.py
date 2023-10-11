import yaml
import polars as pl
import torch
import torch.nn.functional as F
import streamlit as st

# For making dataframes for streamlit
import pandas as pd

from src.net import AstrosNet
from src.data import PitchSequenceDataset


@st.cache_resource
def load_data_and_model(
    config_path,
    morpher_path,
    checkpoint_path,
):
    # This is relative to the root directory, not astros/
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.CLoader)

    # We also need to load the morpher state dicts.
    with open(morpher_path, "r") as f:
        morpher_states = yaml.load(f, yaml.CLoader)

    # Read data
    pitch_df = pl.read_parquet("./data/bigger_training_data.parquet")
    # Filter games with mostly missing pitch data
    pitch_df = pitch_df.filter(
        pl.col("pitch_type").is_null().sum().over("game_pk") <= 30
    )

    # Create the dataset. We'll use the loaded morpher states.
    ds = PitchSequenceDataset(
        pitch_df,
        feature_config=config["features"],
        sequence_length=config["net_params"]["sequence_length"],
        morpher_states=morpher_states,
    )

    net = AstrosNet.load_from_checkpoint(
        checkpoint_path,
        **config["net_params"],
        optim_lr=config["training_params"]["learning_rate"],
        morphers=ds.morphers,
        strict=False,
    )

    return config, ds, net


config, ds, net = load_data_and_model(
    config_path="./astros/astros_config.yaml",
    morpher_path="./reference/morphers.yaml",
    checkpoint_path="./astros/checkpoint/epoch=6-step=15547.ckpt",
)


idx = st.sidebar.number_input("Index", 0, len(ds))
feature = st.sidebar.selectbox(
    "Feature",
    [feat for feat, ftype in config["features"].items() if ftype == "categorical"],
)

x, is_sog = ds[idx]
print(net.device)

print("Things are happening?")

preds = net({k: v.to(net.device) for k, v in x.items()}, is_sog.to(net.device))
print("Seriously wtf")
max_vals, max_idx = torch.max(F.softmax(preds[feature], dim=-1), dim=-1)

df = pd.DataFrame(
    {
        "Ground Truth": x[feature].squeeze()[1:].cpu().detach().numpy(),
        "Prediction": max_idx.squeeze()[:-1].cpu().detach().numpy(),
        "Probability": max_vals.squeeze()[:-1].cpu().detach().numpy(),
        "Pitcher": x["pitcher"].squeeze()[1:].cpu().detach().numpy(),
        "Batter": x["batter"].squeeze()[1:].cpu().detach().numpy(),
    }
)

st.table(data=df)
