import polars as pl
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader, Subset

import streamlit as st

from src.net import AstrosNet
from src.data import PitchSequenceDataset

# Why is it like this
st.set_page_config(layout="wide")

config_path = "./astros/astros_config.yaml"
morpher_path = "./reference/morphers.yaml"
checkpoint_path = "./astros/checkpoint/epoch=6-step=15547.ckpt"

print("Loading everything")

pitch_glossary = {
    "AB": "Automatic Ball",
    "AS": "Automatic Strike",
    "CH": "Change-up",
    "CU": "Curveball",
    "CS": "Slow Curve",
    "EP": "Eephus",
    "FA": "Fastball (Position Player)",
    "FC": "Cutter",
    "FF": "Four-Seam Fastball",
    "FO": "Forkball",
    "FS": "Splitter",
    "FT": "Two-Seam Fastball",
    "GY": "Gyroball",
    "IN": "Intentional Ball",
    "KC": "Knuckle Curve",
    "KN": "Knuckleball",
    "NP": "No Pitch",
    "PO": "Pitchout",
    "SC": "Screwball",
    "SI": "Sinker",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "UN": "Unknown",
}


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

    name_df = pitch_df.select(["pitcher", "player_name"]).unique()
    name_dict = {
        pitcher: player_name
        for pitcher, player_name in zip(name_df["pitcher"], name_df["player_name"])
    }

    print("Creating dataset")

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
        checkpoint_path,
        **config["net_params"],
        optim_lr=config["training_params"]["learning_rate"],
        morphers=ds.dataset.morphers,
        strict=False,
    )

    return config, ds, dl, name_dict, net


@st.cache_data
def score_data(_net, _dl, limit=10):
    actual_pitchers = []
    actual_pitches = []
    predicted_pitches = []

    _net.eval()

    with torch.inference_mode():
        for i, (x, is_sog) in enumerate(_dl):
            actual_pitchers += x["pitcher"]
            actual_pitches += x["pitch_type"]

            preds = _net(
                {k: v.to(_net.device) for k, v in x.items()}, is_sog.to(_net.device)
            )
            predicted_pitches += preds["pitch_type"]

            if i == (limit - 1):
                break

    actual_pitchers = torch.cat(actual_pitchers, dim=0)
    actual_pitches = torch.cat(actual_pitches, dim=0)
    predicted_pitches = torch.nn.functional.softmax(
        torch.cat(predicted_pitches, dim=0), dim=-1
    )

    result_df = pl.DataFrame(
        {
            "actual_pitcher": actual_pitchers.cpu().numpy(),
            "actual_pitch": actual_pitches.cpu().numpy(),
            "predicted_pitch": predicted_pitches.cpu().numpy(),
        }
    )

    return (
        result_df.with_columns(pl.col("predicted_pitch").list.to_struct())
        .unnest("predicted_pitch")
        .with_columns(**{f"actual_{i}": pl.col("actual_pitch") == i for i in range(18)})
        .groupby(pl.col("actual_pitcher"))
        .sum()
        .with_columns(pitcher_name=pl.col("actual_pitcher").map_dict(idx_to_name))
        .drop(["actual_pitch"])
    )


config, ds, dl, name_dict, net = load_data_and_model(
    config_path,
    morpher_path,
    checkpoint_path,
)

pitch_name_to_idx = {
    pitch_glossary[pitch_code]: idx
    for pitch_code, idx in ds.dataset.morphers["pitch_type"].vocab.items()
}
idx_to_pitch_name = {v: k for k, v in pitch_name_to_idx.items()}

# So we can turn pitcher names into indices
name_to_idx = {
    name: ds.dataset.morphers["pitcher"].vocab[pid] for pid, name in name_dict.items()
}
idx_to_name = {v: k for k, v in name_to_idx.items()}

# Get a pitcher
sp = st.selectbox("Pitcher", sorted(name_to_idx.keys()))
sp_idx = name_to_idx[sp]

result_df = score_data(net, dl, limit=20)

pd_df = result_df.filter(pl.col("actual_pitcher") == sp_idx).to_pandas()
col_names = [idx_to_pitch_name[idx] for idx in range(17)]
real_df = pd_df[[f"actual_{i}" for i in range(17)]]
real_df.columns = col_names
pred_df = pd_df[[f"field_{i}" for i in range(17)]]
pred_df.columns = col_names
display_df = pd.concat([real_df, pred_df], ignore_index=True).transpose()
display_df.columns = ["Actual", "Predicted"]

st.dataframe(display_df, height=640)
