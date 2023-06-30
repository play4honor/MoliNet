import torch
from torch.utils.data import Dataset


class CategoricalIndexer:
    """
    This is stolen from Deep-Baseline
    """

    def __init__(self):
        self.category_dictionary = dict()
        self.reverse_dictionary = dict()

    def __getitem__(self, x):
        if x in self.category_dictionary:
            return self.category_dictionary[x]

        else:
            new_idx = len(self.category_dictionary)

            self.category_dictionary[x] = new_idx
            self.reverse_dictionary[new_idx] = x

            return new_idx

    def inverse(self, idx):
        return self.reverse_dictionary[idx]


PITCH_GLOSSARY = {
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

RESULT_GLOSSARY = {"S": "strike or foul", "B": "ball or hbp", "X": "in play"}

# not using CategoricalIndexer for these because they're predefined from Statcast
PITCH_VOCAB_IDX_TOK = {i: k for i, k in enumerate(PITCH_GLOSSARY.keys())}

PITCH_VOCAB_TOK_IDX = {v: k for k, v in PITCH_VOCAB_IDX_TOK.items()}

RESULT_VOCAB_IDX_TOK = {i: k for i, k in enumerate(RESULT_GLOSSARY.keys())}

RESULT_VOCAB_TOK_IDX = {v: k for k, v in RESULT_VOCAB_IDX_TOK.items()}


def clean_statcast(statcast_data):
    statcast_data = statcast_data.sort_values(["game_pk", "inning", "at_bat_number"])

    # Unique at-bat IDs as index
    statcast_data["at_bat_id"] = statcast_data.groupby(
        ["pitcher", "batter", "game_pk", "at_bat_number"]
    ).ngroup()
    statcast_data = statcast_data.set_index("at_bat_id")

    # TKTK: More cleaning as needed

    return statcast_data


class PitchBaseDataset(Dataset):
    """
    This class does pre-processing for pitch sequences from statcast data and
    produces a basic sequence of pitches with no additional information and no
    output label. This constructs the sequence on the fly, which may be stupid
    """

    def __init__(self, statcast_data):
        assert (
            statcast_data.index.name == "at_bat_id"
        ), "statcast_data must have at_bat_id as index"
        self.data = statcast_data

    def __getitem__(self, x):
        ab = self.data.loc[x]
        pitches = torch.tensor(
            ab.pitch_type.map(PITCH_VOCAB_TOK_IDX).to_list(), dtype=torch.long
        )
        results = torch.tensor(
            ab.type.map(RESULT_VOCAB_TOK_IDX).to_list(), dtype=torch.long
        )

        return {"pitches": pitches, "results": results}

    def __len__(self):
        return len(self.data.at_bat_id.unique())


# TKTK: Return stuff for game situation, batter/pitcher, etc

if __name__ == "__main__":
    from stat_s3 import get_s3_client, get_year_df

    client = get_s3_client(aws_profile="p4h")

    data = get_year_df(client, 2022)

    clean_data = clean_statcast(data)

    dataset = PitchBaseDataset(clean_data)

    print(next(iter(dataset)))
