import polars as pl
import pandas as pd
import yaml
import torch

import streamlit as st

from src.net import AstrosNet
from src.data import PitchSequenceDataset

# Why is it like this
st.set_page_config(layout="wide")


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

    return config, ds, net, name_dict


def get_embedding_distance(target_idx, embeddings: torch.nn.Module):
    pitcher_vector = embeddings[target_idx]

    d = torch.nn.functional.cosine_similarity(pitcher_vector, embeddings, dim=1)
    close_idx = torch.argsort(d, descending=True)[1:11]
    close_similarities = d[close_idx]

    return close_idx, close_similarities


config, ds, net, name_dict = load_data_and_model(
    config_path="./astros/astros_config.yaml",
    morpher_path="./reference/morphers.yaml",
    checkpoint_path="./astros/checkpoint/epoch=6-step=15547.ckpt",
)

# So we can turn pitcher names into indices
name_to_idx = {
    name: ds.morphers["pitcher"].vocab[pid] for pid, name in name_dict.items()
}
idx_to_name = {v: k for k, v in name_to_idx.items()}
pitcher_embeddings = net.net.feature_embedder["pitcher"][0].weight

# Get a pitcher
sp = st.selectbox("Pitcher", sorted(name_to_idx.keys()))
sp_idx = name_to_idx[sp]

# Get similar pitchers and the pitcher's pitch mix
similar_idx, similar_sims = get_embedding_distance(sp_idx, pitcher_embeddings)

similar_names = [idx_to_name[idx.item()] for idx in similar_idx]
similar_sims = similar_sims.detach().cpu().numpy()

similar_df = pd.DataFrame(
    {
        "Name": similar_names,
        "Cosine Similarity": similar_sims,
    }
)

st.header("Similar Pitchers")
st.markdown("Similarity measured using cosine similarity on embeddings")
st.dataframe(data=similar_df, hide_index=True)
