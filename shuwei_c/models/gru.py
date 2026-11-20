from __future__ import annotations

import torch
from torch import nn

from .outputs import ForecastOutput


class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, target_dim: int, hidden_dim=96, num_layers=2, dropout=0.15, probabilistic=True):
        super().__init__()
        self.horizon, self.target_dim, self.probabilistic = horizon, target_dim, probabilistic
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        output_dim = horizon * target_dim * (2 if probabilistic else 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        _, hidden = self.encoder(x)
        raw = self.head(hidden[-1])
        if not self.probabilistic:
            return ForecastOutput(raw.view(-1, self.horizon, self.target_dim))
        mean, log_var = raw.chunk(2, dim=-1)
        return ForecastOutput(mean.view(-1, self.horizon, self.target_dim), log_var.view(-1, self.horizon, self.target_dim).clamp(-10, 6))

