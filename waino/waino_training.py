import pandas as pd
import yaml

from torch.utils.data import DataLoader, Subset
import torch

from lightning.pytorch import Trainer

from src.data import PitchSequenceDataset
from src.net import Waino

import random

# Setup config ------------------

print("Reading config")
# This is relative to the root directory, not waino/
with open("./waino/waino_config.yaml", "r") as f:
    config = yaml.load(f, yaml.CLoader)

# Setup data --------------------

print(f"Loading and transforming data from {config['data']}")

pitch_df = pd.read_parquet(config["data"])

ds = PitchSequenceDataset(
    pitch_df,
    min_length=config["training_params"]["min_length"],
    max_length=config["net_params"]["max_length"],
    p_mask=config["training_params"]["p_mask"],
    mask_tokens=config["training_params"]["mask_tokens"],
)
ds.save_vocab("./reference/vocab.yaml")

n_val = int(len(ds) * config["training_params"]["holdout_prob"])
permuted_idx = list(range(len(ds)))
random.shuffle(permuted_idx)
train_idx = permuted_idx[n_val:]
validation_idx = permuted_idx[:n_val]

train_ds = Subset(ds, train_idx)
validation_ds = Subset(ds, validation_idx)

train_dl = DataLoader(
    train_ds, batch_size=config["training_params"]["batch_size"], shuffle=True
)
validation_dl = DataLoader(
    validation_ds, batch_size=config["training_params"]["batch_size"], shuffle=True
)

print(
    f"Loaded {len(ds)} sequences as {len(train_dl)} training batches and {len(validation_dl)} validation batches."
)

# Setup network -----------------

net = Waino(
    **config["net_params"],
    n_tokens=len(ds.get_vocab()),
    optim_lr=config["training_params"]["learning_rate"],
    mask_tokens=config["training_params"]["mask_tokens"],
)

print(f"Created model with {sum(p.numel() for p in net.parameters())} weights.")

# Train the model ---------------

if __name__ == "__main__":
    trainer = Trainer(
        max_epochs=config["training_params"]["max_epochs"],
        log_every_n_steps=config["training_params"]["log_every_n"],
        accumulate_grad_batches=config["training_params"]["accumulate_grad_batches"],
        precision="16-mixed",
    )

    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=validation_dl)
