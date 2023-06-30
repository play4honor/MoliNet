import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import json


class PitchSequenceDataset(Dataset):
    def __init__(
        self,
        pitch_df,
        vocab=None,
        p_mask: float = 0.2,
        max_length: int = 64,
        min_length: int = 0,
        mask_tokens: bool = False,
    ):
        self.pitcher_games = (
            pitch_df.fillna(
                {
                    "pitch_type": "MISSING",
                }
            )
            .sort_values(["pitch_number"])  # This order's each team's pitches in a game
            .groupby(
                ["pitcher", "batter", "game_pk", "at_bat_number"]
            )  # to get per pitcher-game
            .agg({"pitch_type": list})
            .reset_index()
            .sort_values("at_bat_number")
            .groupby(["pitcher", "game_pk"])
            .apply(self._make_pitch_sequence)
            .reset_index()
        )

        # Remove very short appearances, because it seems like they can cause
        # issues.
        if min_length > 0:
            allowed_appearances = (
                self.pitcher_games.pitch_sequence.apply(len) >= min_length
            )
            print(f"Removing {(~allowed_appearances).sum()} short appearances.")
            self.pitcher_games = self.pitcher_games[allowed_appearances]

        self.vocab = self._make_vocab(pitch_df, mask_tokens) if vocab is None else vocab

        self.p_mask = p_mask
        self.max_length = max_length

        # Just apply the vocab in advance
        self.pitcher_games.pitch_sequence = self.pitcher_games.pitch_sequence.map(
            lambda x: [self.vocab[z] for z in x]
        )

    def _make_pitch_sequence(self, df):
        """This is pretty slow."""

        out_sequence = []

        for row in df.itertuples():
            out_sequence += ["<SEP>"]
            out_sequence += [f"B_{row.batter}"]
            out_sequence += row.pitch_type
            pitcher = (
                row.pitcher
            )  # This is duplicative, because we only need it once, but.

        out_sequence = [f"P_{pitcher}"] + out_sequence

        return pd.DataFrame({"pitch_sequence": [out_sequence]})

    def _make_vocab(self, df, mask_tokens):
        pdf = df.fillna({"pitch_type": "MISSING"})
        pitches = pdf.pitch_type.unique().tolist()

        pitchers = [f"P_{p}" for p in pdf.pitcher.unique().tolist()]
        batters = [f"B_{b}" for b in pdf.batter.unique().tolist()]

        all_tokens = pitches + pitchers + batters + ["<SEP>"]

        print(f"Created vocab:")
        print(f"{len(pitches)} pitch types")
        print(f"{len(pitchers)} pitchers")
        print(f"{len(batters)} batters")

        vocab = {t: i for i, t in enumerate(all_tokens)}

        if mask_tokens:
            print("Adding mask token")
            vocab["<MASK>"] = len(vocab)

        return vocab

    def save_vocab(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    def __getitem__(self, idx):
        x = torch.tensor(self.pitcher_games.iloc[idx].pitch_sequence, dtype=torch.long)

        # Remove anything over the maximum size
        x = x[: min(self.max_length, x.shape[0])]

        x_length = x.shape[0]
        right_pad_length = self.max_length - x_length

        x = F.pad(x, (0, right_pad_length), value=0)

        # Create a mask that indicates the padding positions.
        padding_mask = torch.zeros([self.max_length], dtype=torch.bool)
        padding_mask[x_length:] = True

        # I need do-while, lol
        remake_mask = True

        # If the mask would mask every position, generate a different one.
        while remake_mask:
            pretrain_mask = torch.rand_like(x, dtype=torch.float) < self.p_mask
            remake_mask = pretrain_mask[:x_length].all()

        return x, padding_mask, pretrain_mask

    def __len__(self):
        return len(self.pitcher_games)

    def get_vocab(self):
        return self.vocab
