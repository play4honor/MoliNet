import polars as pl
import yaml

from torch.utils.data import DataLoader, Subset

from lightning.pytorch import Trainer
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

from src.data import PitchSequenceDataset
from src.net import AstrosNet

import random

# Setup config ------------------

print("Reading config")
# This is relative to the root directory, not astros/
with open("./astros/astros_config.yaml", "r") as f:
    config = yaml.load(f, yaml.CLoader)

# Setup data --------------------

print(f"Loading and transforming data from {config['data']}")

pitch_df = pl.read_parquet(config["data"])
# Filter games with mostly missing pitch data
pitch_df = pitch_df.filter(pl.col("pitch_type").is_null().sum().over("game_pk") <= 30)

ds = PitchSequenceDataset(
    pitch_df,
    feature_config=config["features"],
    sequence_length=config["net_params"]["sequence_length"],
)
ds.save_state("./reference")

n_val = int(len(ds) * config["training_params"]["holdout_prob"])
permuted_idx = list(range(len(ds)))
random.shuffle(permuted_idx)
train_idx = permuted_idx[n_val:]
validation_idx = permuted_idx[:n_val]

train_ds = Subset(ds, train_idx)
validation_ds = Subset(ds, validation_idx)

train_dl = DataLoader(
    train_ds,
    batch_size=config["training_params"]["batch_size"],
    shuffle=True,
    num_workers=0,
)
validation_dl = DataLoader(
    validation_ds,
    batch_size=config["training_params"]["batch_size"],
    shuffle=True,
    num_workers=0,
)

print(
    f"Loaded {len(ds)} sequences as {len(train_dl)} training batches and {len(validation_dl)} validation batches."
)

# Setup network -----------------

net = AstrosNet(
    **config["net_params"],
    optim_lr=config["training_params"]["learning_rate"],
    morphers=train_ds.dataset.morphers,
)

print(f"Created model with {sum(p.numel() for p in net.parameters())} weights.")

# Train the model ---------------

if __name__ == "__main__":
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config["training_params"]["max_epochs"],
        log_every_n_steps=config["training_params"]["log_every_n"],
        accumulate_grad_batches=config["training_params"]["accumulate_grad_batches"],
        precision="16-mixed",
        # profiler=SimpleProfiler(),
        # fast_dev_run=10,
    )

    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=validation_dl)
