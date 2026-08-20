"""Small CPU models that expose key architectural choices."""

from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    choices: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }
    if name not in choices:
        raise ValueError(f"unknown activation: {name}")
    return choices[name]


class MLPClassifier(nn.Module):
    def __init__(
        self, inputs: int, hidden: int = 32, activation_name: str = "relu", norm: str = "layer"
    ):
        super().__init__()
        normalization: nn.Module
        if norm == "batch":
            normalization = nn.BatchNorm1d(hidden)
        elif norm == "layer":
            normalization = nn.LayerNorm(hidden)
        elif norm == "group":
            normalization = nn.GroupNorm(4, hidden)
        elif norm == "none":
            normalization = nn.Identity()
        else:
            raise ValueError(f"unknown normalization: {norm}")
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden),
            normalization,
            activation(activation_name),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SequenceClassifier(nn.Module):
    def __init__(self, cell: str, inputs: int = 3, hidden: int = 24):
        super().__init__()
        cells: dict[str, type[nn.RNNBase]] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if cell not in cells:
            raise ValueError(f"unknown recurrent cell: {cell}")
        self.recurrent = cells[cell](inputs, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, _ = self.recurrent(inputs)
        return self.output(values[:, -1])


class Autoencoder(nn.Module):
    def __init__(self, inputs: int, latent: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, 8), nn.ReLU(), nn.Linear(8, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 8), nn.ReLU(), nn.Linear(8, inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


def initialize(model: nn.Module, strategy: str) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif strategy == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"unknown initialization: {strategy}")
            nn.init.zeros_(module.bias)
