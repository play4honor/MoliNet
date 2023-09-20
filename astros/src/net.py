import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm

from .helper_layers import BoringPositionalEncoding


class GarbageCan(nn.Module):
    def __init__(
        self,
        d_model: int,
        sequence_length: int,
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
                    nn.GELU(),
                )
                for feature, morpher in morphers.items()
            }
        )

        self.position_encoder = BoringPositionalEncoding(sequence_length, d_model)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead=n_head,
                dim_feedforward=dim_ff,
                batch_first=True,
                activation="gelu",
                dropout=dropout,
            ),
            num_layers=n_layers,
        )
        self.feature_predictors = nn.ModuleDict(
            {
                feature: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    morpher.make_predictor_head(d_model),
                )
                for feature, morpher in morphers.items()
            }
        )

    def forward(self, x_features):
        x = sum(
            [
                embedder(x_features[feature])
                for feature, embedder in self.feature_embedder.items()
            ]
        )

        x = self.position_encoder(x)

        causal_mask = torch.triu(
            torch.ones([x.shape[1], x.shape[1]]).to(x.device).bool(), diagonal=1
        )

        x = self.tr(x, mask=causal_mask, is_causal=True)
        feature_predictions = {
            feature: predictor(x)
            for feature, predictor in self.feature_predictors.items()
        }
        # For pretraining, the last position doesn't have a target to predict
        return feature_predictions


class AstrosNet(pl.LightningModule):
    def __init__(
        self,
        d_model,
        sequence_length,
        n_head,
        n_layers,
        dim_ff,
        dropout,
        optim_lr,
        morphers,
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.net = GarbageCan(
            d_model,
            sequence_length,
            n_head,
            n_layers,
            dim_ff,
            dropout,
            morphers,
        )
        self.optim_lr = optim_lr
        self.feature_criteria = {
            feature: morpher.make_criterion() for feature, morpher in morphers.items()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def step(self, batch, batch_idx):
        x, is_sog = batch

        feature_preds = self.net(x)

        loss_mask = is_sog.bool()

        feat_loss_dict = {
            feat: (
                criterion(
                    feature_preds[feat].squeeze(-1)[:, :-1],
                    x[feat][:, 1:],
                )
                * loss_mask[:, :-1]
            ).sum()
            / (loss_mask[:, :-1].sum())
            for feat, criterion in self.feature_criteria.items()
        }

        # Get per-masked-token loss
        feature_loss = sum(list(feat_loss_dict.values()))

        return feature_loss, feat_loss_dict

    def training_step(self, batch, batch_idx):
        feature_loss, feat_loss_dict = self.step(batch, batch_idx)

        self.log("train_feature_loss", feature_loss)
        self.log_dict({f"train_{k}_loss": v for k, v in feat_loss_dict.items()})

        return feature_loss

    def validation_step(self, batch, batch_idx):
        feature_loss, feat_loss_dict = self.step(batch, batch_idx)

        self.log("valid_feature_loss", feature_loss)
        self.log_dict({f"valid_{k}_loss": v for k, v in feat_loss_dict.items()})

        return feature_loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.net, norm_type=2)
        self.log_dict(norms)
