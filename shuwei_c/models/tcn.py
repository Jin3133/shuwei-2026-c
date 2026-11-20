"""Causal TCN adapted for trajectory regression.

Architecture follows the residual/dilated causal-convolution pattern from
https://github.com/locuslab/TCN (MIT License), rewritten for the local API.
"""
from __future__ import annotations

import torch
from torch import nn

from .outputs import ForecastOutput


class Chomp1d(nn.Module):
    def __init__(self, size: int): super().__init__(); self.size = size
    def forward(self, x): return x[:, :, :-self.size].contiguous() if self.size else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__(); padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation), Chomp1d(padding), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation), Chomp1d(padding), nn.GELU(), nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.GELU()

    def forward(self, x): return self.activation(self.net(x) + self.residual(x))


class TCNForecaster(nn.Module):
    def __init__(self, input_dim, horizon, target_dim, channels=(64, 64, 96), kernel_size=3, dropout=0.15, probabilistic=True):
        super().__init__(); self.horizon, self.target_dim, self.probabilistic = horizon, target_dim, probabilistic
        blocks, current = [], input_dim
        for level, width in enumerate(channels):
            blocks.append(TemporalBlock(current, width, kernel_size, 2**level, dropout)); current = width
        self.encoder = nn.Sequential(*blocks)
        output_dim = horizon * target_dim * (2 if probabilistic else 1)
        self.head = nn.Sequential(nn.LayerNorm(current), nn.Linear(current, output_dim))

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        encoded = self.encoder(x.transpose(1, 2))[:, :, -1]
        raw = self.head(encoded)
        if not self.probabilistic: return ForecastOutput(raw.view(-1, self.horizon, self.target_dim))
        mean, log_var = raw.chunk(2, dim=-1)
        return ForecastOutput(mean.view(-1, self.horizon, self.target_dim), log_var.view(-1, self.horizon, self.target_dim).clamp(-10, 6))

