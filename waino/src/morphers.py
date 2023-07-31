import numpy as np
import torch

from .net_layers import Unsqueezer


class Normalizer:
    def __init__(self, mean, std):
        self.required_dtype = torch.float32

        if isinstance(mean, (np.ndarray, np.generic, torch.Tensor)):
            self.mean = mean.item()
            self.std = std.item()

        else:
            self.mean = mean
            self.std = std

    def normalize(self, x):
        x = (x - self.mean) / self.std
        return np.nan_to_num(x, self.mean)

    def denormalize(self, x):
        # reverse operation
        return x * self.std + self.mean

    def __call__(self, x):
        return self.normalize(x)

    @classmethod
    def from_data(cls, x):
        mean = x.mean()
        std = x.std()

        return cls(mean, std)

    def to_dict(self):
        return {
            "class": "Normalizer",
            "mean": self.mean,
            "std": self.std,
        }

    def __repr__(self):
        return f"Normalizer(mean={self.mean}, std={self.std})"

    def make_embedding(self, x, /):
        return torch.nn.Sequential(
            Unsqueezer(dim=-1),
            torch.nn.Linear(in_features=1, out_features=x),
        )
