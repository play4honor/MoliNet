import polars as pl

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import yaml

import sys
import itertools

sys.path.insert(0, ".")
from src.morphers import Normalizer, Integerizer, BigIntegerizer


class PitchSequenceDataset(Dataset):
    MORPHER_MAP = {
        "numeric": Normalizer,
        "categorical": Integerizer,
        "bigcat": BigIntegerizer,
    }

    def __init__(
        self,
        pitch_df,
        feature_config: dict,
        sequence_length: int = 64,
        morpher_states: dict = None,
    ):
        self.sequence_length = sequence_length

        pitch_df = pitch_df.with_columns(
            (
                pl.col("at_bat_number")
                .rank(method="dense")
                .over(["pitcher", "game_pk"])
                .alias("n_batters_faced")
            )
        )

        numeric_cols = [
            feat for feat, feat_type in feature_config.items() if feat_type == "numeric"
        ]
        pitch_df = pitch_df.with_columns(
            *[pl.col(ncol).cast(pl.Float32) for ncol in numeric_cols]
        )

        # If we have saved morpher states. They still need to match the config.
        if morpher_states is not None:
            print("Using saved morpher states.")
            self.morphers = {
                feature: self.MORPHER_MAP[feature_type].from_state_dict(
                    morpher_states[feature]
                )
                for feature, feature_type in feature_config.items()
            }
        else:
            self.morphers = {
                feature: self.MORPHER_MAP[feature_type].from_data(pitch_df[feature])
                for feature, feature_type in feature_config.items()
            }

        # Get the n_batters_faced feature
        pitches = pitch_df.select(
            *[morpher(pl.col(feature)) for feature, morpher in self.morphers.items()],
            pl.col("game_pk"),
            pl.col("at_bat_number"),
            pl.col("pitch_number"),
        )

        # Sorry
        ordered_games = pitches.sort(["at_bat_number", "pitch_number"]).with_columns(
            pl.lit(False).alias("is_sog")
        )
        sog_row = ordered_games[0].clone().with_columns(is_sog=True)

        grouped_games = ordered_games.groupby("game_pk")

        sog_rows = [sog_row.clone() for _ in grouped_games]
        result_df = pl.concat(
            itertools.chain.from_iterable(zip(sog_rows, [g for _, g in grouped_games]))
        )

        self.pitches = result_df

    def __len__(self):
        return len(self.pitches) - self.sequence_length

    def save_state(self, path):
        with open(f"{path}/morphers.yaml", "w") as f:
            yaml.dump(
                {feat: m.save_state_dict() for feat, m in self.morphers.items()},
                f,
            )

    def __getitem__(self, idx):
        pitch_subset = self.pitches.slice(idx, self.sequence_length)

        return {
            feature: torch.tensor(pitch_subset[feature], dtype=morpher.required_dtype)
            for feature, morpher in self.morphers.items()
        }, torch.tensor(pitch_subset["is_sog"], dtype=torch.bool)


if __name__ == "__main__":
    # This is relative to the root directory, not waino/
    with open("./waino_config.yaml", "r") as f:
        config = yaml.load(f, yaml.CLoader)

    pitch_df = pl.read_parquet("../data/test_data.parquet")

    ds = PitchSequenceDataset(
        pitch_df,
        config["features"],
        sequence_length=config["net_params"]["sequence_length"],
    )

    print(ds[0])
