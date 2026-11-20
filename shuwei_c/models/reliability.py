"""Observation-reliability models inspired by GRU-D's explicit missingness inputs."""
from __future__ import annotations

import torch
from torch import nn


class ReliabilityEstimator(nn.Module):
    """Estimate per-timestep observation reliability from causal history."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        states, _ = self.encoder(evidence)
        return self.head(states).squeeze(-1)


class QualityGatedInput(nn.Module):
    """Learn a causal gate from quality, observation mask and time gap."""

    def __init__(self, value_dim: int, evidence_dim: int, output_dim: int):
        super().__init__()
        self.value_projection = nn.Linear(value_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(evidence_dim, output_dim), nn.Sigmoid())
        self.evidence_projection = nn.Linear(evidence_dim, output_dim)

    def forward(self, values: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        gate = self.gate(evidence)
        return gate * self.value_projection(values) + self.evidence_projection(evidence)

