"""Prefix-conditioned probabilistic duration models."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DurationQuantileOutput:
    def __init__(self, quantiles: torch.Tensor, levels: tuple[float, ...]):
        self.quantiles = quantiles
        self.levels = levels


class PrefixDurationQuantileModel(nn.Module):
    """Causal prefix encoder with structurally non-crossing duration quantiles."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1, levels=(0.5, 0.8, 0.9), min_duration_s: float = 0.2):
        super().__init__()
        if tuple(levels) != tuple(sorted(levels)): raise ValueError("Quantile levels must be sorted")
        self.levels, self.min_duration_s = tuple(levels), min_duration_s
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, len(levels)))

    def forward(self, prefix: torch.Tensor, prefix_mask: torch.Tensor | None = None) -> DurationQuantileOutput:
        states, _ = self.encoder(prefix)
        scores = self.attention(states).squeeze(-1)
        if prefix_mask is not None: scores = scores.masked_fill(~prefix_mask, torch.finfo(scores.dtype).min)
        pooled = torch.sum(states * torch.softmax(scores, dim=1).unsqueeze(-1), dim=1)
        raw = self.head(pooled)
        first = self.min_duration_s + F.softplus(raw[:, :1])
        quantiles = torch.cat([first, first + torch.cumsum(F.softplus(raw[:, 1:]), dim=-1)], dim=-1) if raw.shape[1] > 1 else first
        return DurationQuantileOutput(quantiles, self.levels)


def pinball_loss(output: DurationQuantileOutput, target_s: torch.Tensor) -> torch.Tensor:
    error = target_s[:, None] - output.quantiles
    levels = torch.tensor(output.levels, device=error.device, dtype=error.dtype)[None, :]
    return torch.maximum(levels * error, (levels - 1) * error).mean()

