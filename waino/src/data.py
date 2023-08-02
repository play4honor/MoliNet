import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import yaml

from .morphers import Normalizer


class PitchSequenceDataset(Dataset):
    MORPHER_MAP = {
        "numeric": Normalizer,
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
        pitch_df[numeric_cols] = pitch_df[numeric_cols].astype(np.float32)

        self.morphers = {
            feature: self.MORPHER_MAP[feature_type].from_data(pitch_df[feature])
            for feature, feature_type in feature_config.items()
        }
        self.pitcher_games = (
            pitch_df.fillna(
                {
                    "pitch_type": "MISSING",
                }
            )
            .sort_values(["pitch_number"])  # This order's each team's pitches in a game
            .groupby(["pitcher", "batter", "game_pk", "at_bat_number"])  # to get per PA
            .agg({"pitch_type": list} | {f: list for f in feature_config.keys()})
            .reset_index()
            .sort_values(["at_bat_number"])
        )

        pfeat_cols = list(feature_config.keys())
        self.pitcher_games[pfeat_cols] = self.pitcher_games[pfeat_cols].applymap(
            np.array
        )

        self.pitcher_games["n_batters_faced"] = self.pitcher_games.groupby(
            ["pitcher", "game_pk"]
        ).cumcount()
        max_batters_faced = self.pitcher_games.n_batters_faced.max()

        self.pitcher_games["pitch_sequence"] = self.pitcher_games.apply(
            self._make_pitch_sequence, axis=1
        )

        self.vocab = (
            self._make_vocab(pitch_df, max_batters_faced, mask_tokens)
            if vocab is None
            else vocab
        )

        self.p_mask = p_mask
        self.max_length = max_length

        # Just apply the vocab in advance
        self.pitcher_games.pitch_sequence = self.pitcher_games.pitch_sequence.map(
            lambda x: [self.vocab[z] for z in x]
        )

    def _make_pitch_sequence(self, row):
        return (
            [f"BF_{row.n_batters_faced}"]
            + [f"P_{row.pitcher}"]
            + [f"B_{row.batter}"]
            + row.pitch_type
        )

    def _make_vocab(self, df, max_batters_faced, mask_tokens):
        pdf = df.fillna({"pitch_type": "MISSING"})
        pitches = pdf.pitch_type.unique().tolist()

        pitchers = [f"P_{p}" for p in pdf.pitcher.unique().tolist()]
        batters = [f"B_{b}" for b in pdf.batter.unique().tolist()]

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
            yaml.dump(
                {feat: m.save_state_dict() for feat, m in self.morphers.items()}, f
            )

    def __getitem__(self, idx):
        x = torch.tensor(self.pitcher_games.iloc[idx].pitch_sequence, dtype=torch.long)

        # Remove anything over the maximum size
        x = x[: min(self.max_length, x.shape[0])]

        x_length = x.shape[0]
        right_pad_length = self.max_length - x_length

        x = F.pad(x, (0, right_pad_length), value=0)

        # Transform and pad the pitch features.
        pitch_features = {
            feature: torch.tensor(
                morpher(self.pitcher_games[feature].iloc[idx]),
                dtype=morpher.required_dtype,
            )
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

    pitch_df = pd.read_parquet("./data/test_data.parquet")

    ds = PitchSequenceDataset(
        pitch_df,
        max_length=config["net_params"]["max_length"],
        p_mask=config["training_params"]["p_mask"],
        mask_tokens=config["training_params"]["mask_tokens"],
    )

    print(ds.morphers["release_speed"])

    print(ds.pitcher_games.iloc[0])

    print(ds[0])
