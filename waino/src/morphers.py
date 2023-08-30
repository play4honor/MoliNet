from abc import ABC, abstractmethod

import numpy as np
import torch

from .helper_layers import Unsqueezer


class Morpher(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_data(self):
        raise NotImplementedError

    @abstractmethod
    def make_embedding(self):
        raise NotImplementedError

    @abstractmethod
    def make_predictor_head(self):
        raise NotImplementedError

    @abstractmethod
    def make_criterion(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def required_dtype(self):
        raise NotImplementedError

    @abstractmethod
    def save_state_dict(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError


class Normalizer(Morpher):
    def __init__(self, mean, std):
        if isinstance(mean, (np.ndarray, np.generic, torch.Tensor)):
            self.mean = mean.item()
            self.std = std.item()

        else:
            self.mean = mean
            self.std = std

    @property
    def required_dtype(self):
        return torch.float32

    def normalize(self, x):
        x = np.array(x, dtype="float32")
        x = (x - self.mean) / self.std
        return np.nan_to_num(x, self.mean)

    def denormalize(self, x):
        # reverse operation
        return x * self.std + self.mean

    def __call__(self, x):
        return self.normalize(x)

    @classmethod
    def from_data(cls, x):
        mean = x.mean(where=~np.isnan(x))
        std = x.std(where=~np.isnan(x))

        return cls(mean, std)

    def save_state_dict(self):
        return {
            "class": "Normalizer",
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(mean=state_dict["mean"], std=state_dict["std"])

    def __repr__(self):
        return f"Normalizer(mean={self.mean}, std={self.std})"

    def make_embedding(self, x, /):
        return torch.nn.Sequential(
            Unsqueezer(dim=-1),
            torch.nn.Linear(in_features=1, out_features=x),
        )

    def make_predictor_head(self, x, /):
        return torch.nn.Linear(in_features=x, out_features=1)

    def make_criterion(self):
        return torch.nn.MSELoss(reduction="none")


class Integerizer(Morpher):
    def __init__(self, vocab):
        self.vocab = vocab

    @property
    def required_dtype(self):
        return torch.int64

    def __call__(self, x):
        return [self.vocab.get(item, self.vocab["<MISSING>"]) for item in x]

    @classmethod
    def from_data(cls, x):
        vocab = {
            t: i
            for i, t in enumerate(np.unique(x))
            if not isinstance(t, np.generic) or not np.isnan(t)
        }
        vocab["<MISSING>"] = len(vocab)

        return cls(vocab)

    def save_state_dict(self):
        return self.vocab

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict)

    def __repr__(self):
        return f"Integerizer(<{len(self.vocab)} items>)"

    def make_embedding(self, x, /):
        return torch.nn.Embedding(len(self.vocab), x)

    def make_predictor_head(self, x, /):
        return torch.nn.Linear(in_features=x, out_features=len(self.vocab))

    def make_criterion(self):
        def fixed_ce_loss(input, target):
            return torch.nn.functional.cross_entropy(
                input.permute(0, 2, 1), target, reduction="none"
            )

        return fixed_ce_loss
