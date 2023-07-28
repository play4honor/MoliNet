import pandas as pd
import numpy as np
import yaml
from scipy.spatial.distance import jensenshannon
import torch

from waino.src.net import Waino

import streamlit as st


@st.cache_data
def load_pitch_info(data_path, glossary_path, checkpoint_path, vocab_path):
    with open(glossary_path, "r") as f:
        pitch_glossary = yaml.load(f, Loader=yaml.CLoader)
    net = Waino.load_from_checkpoint(checkpoint_path)
    with open(vocab_path, "r") as f:
        vocab = yaml.load(f, Loader=yaml.CLoader)

    pitch_counts = (
        pd.read_parquet(data_path)
        .groupby(["player_name", "pitcher", "pitch_type"])
        .apply(len)
        .reset_index()
        .rename({0: "pitch_count"}, axis=1)
    )

    # Use readable names for pitch types
    pitch_counts["pitch_type"] = pitch_counts["pitch_type"].map(pitch_glossary)

    # Get total pitches for each pitcher, and fraction
    pitch_counts["total"] = (
        pitch_counts["pitch_count"]
        .groupby(pitch_counts["player_name"])
        .transform("sum")
    )
    pitch_counts["pitch_fraction"] = pitch_counts.pitch_count / pitch_counts.total
    pitch_counts = pitch_counts.drop("total", axis=1)

    pitch_counts.columns = [
        "Pitcher",
        "pitcher_id",
        "Pitch Type",
        "N",
        "Pitch Fraction",
    ]

    pitch_vectors = (
        pitch_counts.pivot(
            columns=["Pitch Type"],
            index=["Pitcher", "pitcher_id"],
            values="Pitch Fraction",
        )
        .fillna(0)
        .reset_index()
    )

    embedding_dict = {
        int(k[2:]): net.net.pitch_embedding[0].weight[v, :].detach().cpu()
        for k, v in vocab.items()
        if k[:2] == "P_"
    }
    pitcher_embeddings = pd.DataFrame(
        {
            "Pitcher": pitch_vectors.Pitcher,
            "embedding": pitch_vectors.pitcher_id.map(embedding_dict),
        }
    )

    return pitch_counts, pitch_vectors, pitcher_embeddings


def get_pitch_distance(target_pitcher, df: pd.DataFrame):
    pitcher_vector = df[df.Pitcher == target_pitcher].iloc[:, 2:].to_numpy()
    all_pitcher_vectors = df.iloc[:, 2:].to_numpy()

    d = jensenshannon(pitcher_vector, all_pitcher_vectors, axis=1)
    distance_order = np.argsort(d)
    similar_pitchers = pd.DataFrame(
        {
            "Pitcher": df.Pitcher[distance_order[1:11]],
            "Distance": d[distance_order[1:11]],
        }
    )
    return similar_pitchers


def get_embedding_distance(target_pitcher, embedding_df: pd.DataFrame):
    pitcher_vector = (
        embedding_df.embedding[embedding_df.Pitcher == target_pitcher]
        .item()
        .unsqueeze(0)
    )
    all_pitcher_vectors = torch.stack(embedding_df.embedding.tolist(), dim=0)

    d = torch.nn.functional.cosine_similarity(
        pitcher_vector, all_pitcher_vectors, dim=1
    )
    distance_order = torch.argsort(-d).numpy()
    similar_pitchers = pd.DataFrame(
        {
            "Pitcher": embedding_df.Pitcher[distance_order[1:11]],
            "Cosine Similarity": d[distance_order[1:11]],
        }
    )
    return similar_pitchers


# Why is it like this
st.set_page_config(layout="wide")

# Setup
pitch_mix, pitch_vectors, pitch_embeddings = load_pitch_info(
    "data/bigger_training_data.parquet",
    "reference/pitch_glossary.yaml",
    "lightning_logs/version_1/checkpoints/epoch=6-step=4144.ckpt",
    "reference/vocab.yaml",
)

# Get a pitcher
sp = st.selectbox("Pitcher", pitch_mix.Pitcher.unique())

# Get similar pitchers and the pitcher's pitch mix
similar_pitchers = get_pitch_distance(sp, pitch_vectors)
similar_pitchers_by_embedding = get_embedding_distance(sp, pitch_embeddings)
pitches = pitch_mix.loc[pitch_mix.Pitcher == sp, ["Pitch Type", "N", "Pitch Fraction"]]

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Pitch Mix")
    st.dataframe(
        data=pitches.sort_values("Pitch Fraction", ascending=False),
        hide_index=True,
        column_config={
            "Pitch Fraction": st.column_config.ProgressColumn(
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
    )

with col2:
    st.header("Similar Pitchers")
    st.markdown("Similarity measured using Jensen-Shannon distance on pitch mix")
    st.dataframe(data=similar_pitchers, hide_index=True)

with col3:
    st.header("Similar Pitchers Part 2")
    st.markdown("Similarity measured using cosine similarity on embeddings")
    st.dataframe(data=similar_pitchers_by_embedding, hide_index=True)
