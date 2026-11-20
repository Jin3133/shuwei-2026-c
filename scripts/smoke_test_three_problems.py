#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.active import UncertaintyTriggerPolicy
from shuwei_c.active.policies import ObservationContext
from shuwei_c.models.duration import PrefixDurationQuantileModel, pinball_loss
from shuwei_c.models.reliability import QualityGatedInput, ReliabilityEstimator


def main():
    batch, length = 8, 20
    evidence = torch.randn(batch, length, 4)
    reliability = ReliabilityEstimator(4)(evidence)
    assert reliability.shape == (batch, length)
    gated = QualityGatedInput(6, 4, 16)(torch.randn(batch, length, 6), evidence)
    assert gated.shape == (batch, length, 16)

    variance = torch.full((12, 3), 0.01)
    decision = UncertaintyTriggerPolicy(0.12).decide(variance, ObservationContext(0.5, False, 0, 4, 0.2))
    assert decision.request

    duration_model = PrefixDurationQuantileModel(input_dim=10)
    output = duration_model(torch.randn(batch, length, 10), torch.ones(batch, length, dtype=torch.bool))
    assert output.quantiles.shape == (batch, 3)
    assert torch.all(output.quantiles[:, 1:] >= output.quantiles[:, :-1])
    loss = pinball_loss(output, torch.rand(batch) * 10 + 1)
    loss.backward(); assert torch.isfinite(loss)
    print("Q1 reliability OK", reliability.shape)
    print("Q2 decision OK", decision)
    print("Q3 duration quantiles OK", output.quantiles.shape, f"loss={loss.item():.6f}")


if __name__ == "__main__": main()
