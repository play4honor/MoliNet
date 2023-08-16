import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
import torchmetrics

import math

from .helper_layers import PositionEncoding


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
        morphers,
    ):
        super().__init__()

        self.n_head = n_head

        self.feature_embedder = nn.ModuleDict(
            {
                feature: nn.Sequential(
                    morpher.make_embedding(d_model),
                    nn.LayerNorm(d_model),
                    nn.LeakyReLU(),
                )
                for feature, morpher in morphers.items()
            }
        )

        self.pitch_embedder = nn.Sequential(
            nn.Embedding(
                math.ceil((n_pitches + 1) / 8) * 8,
                d_model,
            ),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
        )
        self.position_encoder = PositionEncoding(max_length, d_model)
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
        self.token_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, n_pitches),
        )
        self.feature_predictors = nn.ModuleDict(
            {
                feature: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.LeakyReLU(),
                    morpher.make_predictor_head(d_model),
                )
                for feature, morpher in morphers.items()
            }
        )

    def forward(self, x, x_features, pad_mask, pretrain_mask=None):
        feature_embedding = sum(
            [
                embedder(x_features[feature])
                for feature, embedder in self.feature_embedder.items()
            ]
        )
        # TKTK Deal with magic number
        # Mask features by setting feature embeddings at masked positions to 0
        if pretrain_mask is not None:
            feature_embedding[pretrain_mask[:, 3:]] = 0
        x = self.pitch_embedder(x)
        x[:, 3:, :] += feature_embedding
        x = self.position_encoder(x)
        x = self.tr(x, src_key_padding_mask=pad_mask)
        token_predictions = self.token_predictor(x)
        feature_predictions = {
            feature: predictor(x)[:, 3:]
            for feature, predictor in self.feature_predictors.items()
        }
        return token_predictions, feature_predictions


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
        morphers,
        compile_model=False,
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.net = WainoNet(
            n_tokens,
            d_model,
            max_length,
            n_head,
            n_layers,
            dim_ff,
            dropout,
            morphers,
        )
        if compile_model:
            self.net = torch.compile(self.net, backend="inductor")
        self.optim_lr = optim_lr
        self.n_tokens = n_tokens
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.feature_criteria = {
            feature: morpher.make_criterion() for feature, morpher in morphers.items()
        }

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

    def forward(self, x, x_features, pad_map, pretrain_mask):
        # Literally this only gets used to make a graph, wtf

        return self.net(x, x_features, pad_map, pretrain_mask)

    def step(self, batch, batch_idx):
        x, x_features, pad_mask, pretrain_mask = batch

        if self.mask_tokens:
            # For debugging. I'm literally going to squash the actual values in
            # the input to make sure the attention masking is actually working.
            # If it is, this shouldn't matter.
            x2 = x.clone()
            x2[pretrain_mask] = self.mask_idx
            # Features are masked in forward.
            token_preds, feature_preds = self.net(
                x2, x_features, pad_mask, pretrain_mask
            )
        else:
            token_preds, feature_preds = self.net(
                x, x_features, pad_mask, pretrain_mask
            )

        loss_mask = pretrain_mask & (~pad_mask)

        token_loss = self.criterion(
            token_preds.reshape([-1, token_preds.shape[-1]]), x.reshape([-1])
        )
        try:
            feat_loss_dict = {
                feat: (
                    criterion(
                        a := feature_preds[feat].squeeze(-1),
                        b := x_features[feat],
                    )
                    * loss_mask[:, 3:]
                ).sum()
                / (loss_mask[:, 3:].sum())
                for feat, criterion in self.feature_criteria.items()
            }
        except Exception as e:
            print(a.shape)
            print(b.shape)
            raise e

        # Get per-masked-token loss
        token_loss = (token_loss * loss_mask.reshape(-1)).sum() / (loss_mask.sum())
        feature_loss = sum([loss for loss in feat_loss_dict.values()])

        return token_loss, feature_loss, feat_loss_dict

    def training_step(self, batch, batch_idx):
        token_loss, feature_loss, feat_loss_dict = self.step(batch, batch_idx)

        self.log("train_token_loss", token_loss)
        self.log("train_feature_loss", feature_loss)
        self.log_dict({f"train_{k}_loss": v for k, v in feat_loss_dict.items()})

        loss = token_loss + feature_loss

        return loss

    def validation_step(self, batch, batch_idx):
        token_loss, feature_loss, feat_loss_dict = self.step(batch, batch_idx)

        self.log("valid_token_loss", token_loss)
        self.log("valid_feature_loss", feature_loss)
        self.log_dict({f"valid_{k}_loss": v for k, v in feat_loss_dict.items()})

        loss = token_loss + feature_loss

        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.net, norm_type=2)
        self.log_dict(norms)
