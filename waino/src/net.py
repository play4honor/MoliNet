import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
import torchmetrics

import math


# I'm going to learn additive position encodings like a barbarian.
class PositionEncoding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        pe = torch.normal(torch.zeros([1, max_length, d_model]), 1)
        self.position_encodings = nn.Parameter(pe)

    def forward(self, x):
        x_len = x.shape[1]

        return x + self.position_encodings[:, :x_len]


# Design here is:
# - embedding layer
# - position encoding
# - transformer encoder
# - pitch type predictor
class WainoNet(nn.Module):
    def __init__(
        self,
        n_pitches: int,
        d_model: int,
        max_length: int,
        n_head,
        n_layers,
        dim_ff,
        dropout,
    ):
        super().__init__()

        self.n_head = n_head

        self.pitch_embedding = nn.Sequential(
            nn.Embedding(
                math.ceil((n_pitches + 1) / 8) * 8,
                d_model,
            ),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
        )
        self.position_encoding = PositionEncoding(max_length, d_model)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead=n_head,
                dim_feedforward=dim_ff,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=n_layers,
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, n_pitches),
        )

    def forward(self, x, pad_mask):
        x = self.pitch_embedding(x)
        x = self.position_encoding(x)
        x = self.tr(x, src_key_padding_mask=pad_mask)
        x = self.output_layer(x)
        return x


class Waino(pl.LightningModule):
    def __init__(
        self,
        n_tokens,
        d_model,
        max_length,
        n_head,
        n_layers,
        dim_ff,
        dropout,
        optim_lr,
        mask_tokens,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = WainoNet(
            n_tokens,
            d_model,
            max_length,
            n_head,
            n_layers,
            dim_ff,
            dropout,
        )
        self.optim_lr = optim_lr
        self.n_tokens = n_tokens
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.mask_tokens = mask_tokens
        if mask_tokens:
            # The mask token index will always be the last one in the vocab.
            self.mask_idx = n_tokens
            print("Please note that the model has `mask_tokens` set to true.")

        self.metrics = nn.ParameterDict(
            {
                "train_accuracy": torchmetrics.classification.MulticlassAccuracy(
                    n_tokens, average="weighted"
                ),
                "validation_accuracy": torchmetrics.classification.MulticlassAccuracy(
                    n_tokens, average="weighted"
                ),
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(self, x, pad_map):
        # Literally this only gets used to make a graph, wtf

        return self.net(x, pad_map)

    def step(self, batch, batch_idx):
        x, pad_mask, pretrain_mask = batch

        if self.mask_tokens:
            # For debugging. I'm literally going to squash the actual values in
            # the input to make sure the attention masking is actually working.
            # If it is, this shouldn't matter.
            x2 = x.clone()
            x2[pretrain_mask] = self.mask_idx
            y_hat = self.net(x2, pad_mask)
        else:
            y_hat = self.net(x, pad_mask)

        loss_mask = pretrain_mask & (~pad_mask)

        loss = self.criterion(y_hat.reshape([-1, y_hat.shape[-1]]), x.reshape([-1]))

        # Get per-masked-token loss
        loss = (loss * loss_mask.reshape(-1)).sum() / (loss_mask.sum())

        masked_x = x[loss_mask].reshape(-1)
        masked_y_hat = y_hat[loss_mask].reshape(-1, y_hat.shape[-1])

        return masked_x, masked_y_hat, loss

    def training_step(self, batch, batch_idx):
        masked_x, masked_y_hat, loss = self.step(batch, batch_idx)

        self.log("train_loss", loss)
        self.log_dict(
            {
                name: metric(masked_y_hat, masked_x)
                for name, metric in self.metrics.items()
                if "train" in name
            }
        )

        return loss

    def validation_step(self, batch, batch_idx):
        masked_x, masked_y_hat, loss = self.step(batch, batch_idx)

        self.log("validation_loss", loss)
        self.log_dict(
            {
                name: metric(masked_y_hat, masked_x)
                for name, metric in self.metrics.items()
                if "validation" in name
            }
        )

        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.net, norm_type=2)
        self.log_dict(norms)
