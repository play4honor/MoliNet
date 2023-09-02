import polars as pl

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import yaml

from .morphers import Normalizer, Integerizer, QuantileEmbedder


class PitchSequenceDataset(Dataset):
    MORPHER_MAP = {
        "numeric": QuantileEmbedder,
        "categorical": Integerizer,
    }

    def __init__(
        self,
        pitch_df,
        feature_config: dict,
        vocab=None,
        p_mask: float = 0.2,
        max_length: int = 64,
        mask_tokens: bool = False,
    ):
        numeric_cols = [
            feat for feat, feat_type in feature_config.items() if feat_type == "numeric"
        ]
        pitch_df = pitch_df.with_columns(
            *[pl.col(ncol).cast(pl.Float32) for ncol in numeric_cols]
        )

        max_batters_faced = pitch_df.select(
            pl.n_unique("at_bat_number").over(["game_pk", "pitcher"]).max()
        ).item()

        self.vocab = (
            self._make_vocab(pitch_df, max_batters_faced, mask_tokens)
            if vocab is None
            else vocab
        )

        self.morphers = {
            feature: self.MORPHER_MAP[feature_type].from_data(
                pitch_df[feature].to_numpy()
            )
            for feature, feature_type in feature_config.items()
        }

        self.pitcher_games = (
            pitch_df.with_columns(
                pl.col("pitch_type")
                .fill_null("MISSING")
                .map_dict(self.vocab, return_dtype=pl.Int32),
                pl.col("at_bat_number")
                .cumcount()
                .over(["pitcher", "game_pk"])
                .str.replace(r"(.+)", r"BF_$1")
                .map_dict(self.vocab, return_dtype=pl.Int32)
                .alias("n_batters_faced"),
                pl.col("pitcher")
                .str.replace(r"(.+)", r"P_$1")
                .map_dict(self.vocab, return_dtype=pl.Int32),
                pl.col("batter")
                .str.replace(r"(.+)", r"B_$1")
                .map_dict(self.vocab, return_dtype=pl.Int32),
            )
            .groupby(["pitcher", "batter", "game_pk", "at_bat_number"])
            .agg(
                *[
                    pl.col(fcol).sort_by(pl.col("pitch_number"))
                    for fcol in ["pitch_type"] + list(feature_config.keys())
                ]
            )
            .sort(pl.col("at_bat_number"))
        )

        self.pitcher_games = self.pitcher_games.with_columns(
            pl.col("at_bat_number")
            .cumcount()
            .over(["pitcher", "game_pk"])
            .alias("n_batters_faced")
        )

        self.pitcher_games = self.pitcher_games.with_columns(
            pl.concat_list(
                pl.col("n_batters_faced"),
                pl.col("pitcher"),
                pl.col("batter"),
                pl.col("pitch_type"),
            ).alias("pitch_sequence")
        )

        self.p_mask = p_mask
        self.max_length = max_length

    def _make_vocab(self, df, max_batters_faced, mask_tokens):
        pdf = df.with_columns(pl.col("pitch_type").fill_null("MISSING"))
        pitches = pdf["pitch_type"].unique().to_list()

        pitchers = [f"P_{p}" for p in pdf["pitcher"].unique().to_list()]
        batters = [f"B_{b}" for b in pdf["batter"].unique().to_list()]

        batters_faced_tokens = [f"BF_{i}" for i in range(max_batters_faced + 1)]

        all_tokens = sum(
            [
                pitches,
                pitchers,
                batters,
                batters_faced_tokens,
                ["<SEP>"],
            ],
            start=[],
        )

        print(f"Created vocab:")
        print(f"{len(pitches)} pitch types")
        print(f"{len(pitchers)} pitchers")
        print(f"{len(batters)} batters")

        vocab = {t: i for i, t in enumerate(all_tokens)}

        if mask_tokens:
            print("Adding mask token")
            vocab["<MASK>"] = len(vocab)

        return vocab

    def save_state(self, path):
        with open(f"{path}/vocab.yaml", "w") as f:
            yaml.dump(self.vocab, f)

        with open(f"{path}/morphers.yaml", "w") as f:
            try:
                yaml.dump(
                    {
                        feat: (a := m.save_state_dict())
                        for feat, m in self.morphers.items()
                    },
                    f,
                )
            except Exception as e:
                print(a)
                raise e

    def __getitem__(self, idx):
        row = self.pitcher_games.row(idx, named=True)
        x = torch.tensor(row["pitch_sequence"], dtype=torch.long)

        # Remove anything over the maximum size
        x = x[: min(self.max_length, x.shape[0])]

        x_length = x.shape[0]
        right_pad_length = self.max_length - x_length

        x = F.pad(x, (0, right_pad_length), value=0)

        # Transform and pad the pitch features.
        pitch_features = {
            feature: torch.tensor(morpher(row[feature]), dtype=morpher.required_dtype)
            for feature, morpher in self.morphers.items()
        }

        pitch_features = {
            feature: F.pad(x, (0, right_pad_length), value=0)
            for feature, x in pitch_features.items()
        }

        # Create a mask that indicates the padding positions.
        padding_mask = torch.zeros([self.max_length], dtype=torch.bool)
        padding_mask[x_length:] = True

        # I need do-while, lol
        remake_mask = True

        # If the mask would mask every position, generate a different one.
        while remake_mask:
            pretrain_mask = torch.rand_like(x, dtype=torch.float) < self.p_mask
            remake_mask = pretrain_mask[:x_length].all()

        return x, pitch_features, padding_mask, pretrain_mask

    def __len__(self):
        return len(self.pitcher_games)

    def get_vocab(self):
        return self.vocab


if __name__ == "__main__":
    # This is relative to the root directory, not waino/
    with open("./waino/waino_config.yaml", "r") as f:
        config = yaml.load(f, yaml.CLoader)

    pitch_df = pl.read_parquet("./data/test_data.parquet")

    ds = PitchSequenceDataset(
        pitch_df,
        max_length=config["net_params"]["max_length"],
        p_mask=config["training_params"]["p_mask"],
        mask_tokens=config["training_params"]["mask_tokens"],
    )

    print(ds.morphers["release_speed"])

    print(ds.pitcher_games.iloc[0])

    print(ds[0])
