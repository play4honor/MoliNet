import torch
from torch import nn


# I'm going to learn additive position encodings like a barbarian.
class PositionEncoding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        pe = torch.normal(torch.zeros([1, max_length, d_model]), 1)
        self.position_encodings = nn.Parameter(pe)

    def forward(self, x):
        x_len = x.shape[1]
        return x + self.position_encodings[:, :x_len]


# We need this horrible thing so Morphers can use it to return the right shape.
class Unsqueezer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class QuantileEmbedding(torch.nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer("quantiles", quantiles.unsqueeze(0))
        self.output_size = self.quantiles.numel() - 1

    def forward(self, x):
        # x should be n x 1, self.quantiles should be 1 x k
        x = x - self.quantiles[:, :-1]
        x = x / (self.quantiles[:, 1:] - self.quantiles[:, :-1])
        x = torch.clamp(x, 0, 1)
        return x
