"""Joint reliability estimation and probabilistic trajectory forecasting."""
from __future__ import annotations

import torch
from torch import nn

from .outputs import ForecastOutput
from .reliability import QualityGatedInput, ReliabilityEstimator


class QualityAwareGRUForecaster(nn.Module):
    """Main Q1 model: causal quality gate, reliability head and Gaussian forecast.

    Canonical input fields:
      0:6  = relative position and causal velocity
      6    = observed mask (1 observed, 0 missing)
      7    = time since last valid observation in seconds
      8    = raw quality evidence (1 good, 0 bad/unknown)
      9    = hand side indicator
    """

    def __init__(self, input_dim: int, horizon: int, target_dim: int = 3, hidden_dim: int = 96, num_layers: int = 2, dropout: float = 0.15, value_dim: int = 6, evidence_indices=(6, 7, 8, 9)):
        super().__init__()
        if input_dim <= max(evidence_indices): raise ValueError("input_dim does not contain configured evidence fields")
        self.horizon, self.target_dim = horizon, target_dim
        self.value_dim, self.evidence_indices = value_dim, tuple(evidence_indices)
        evidence_dim = len(self.evidence_indices)
        self.input_gate = QualityGatedInput(value_dim, evidence_dim, hidden_dim)
        self.reliability = ReliabilityEstimator(evidence_dim, hidden_dim // 2, 1, dropout)
        self.encoder = nn.GRU(hidden_dim + 1, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, horizon * target_dim * 2))

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        values = x[:, :, : self.value_dim]
        evidence = x[:, :, list(self.evidence_indices)]
        reliability_logits = self.reliability(evidence)
        gated = self.input_gate(values, evidence)
        encoded_input = torch.cat([gated, reliability_logits.sigmoid().unsqueeze(-1)], dim=-1)
        _, hidden = self.encoder(encoded_input)
        mean, log_variance = self.head(hidden[-1]).chunk(2, dim=-1)
        return ForecastOutput(
            mean.view(-1, self.horizon, self.target_dim),
            log_variance.view(-1, self.horizon, self.target_dim).clamp(-10, 6),
            reliability_logits,
        )
