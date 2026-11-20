#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.calibration import HorizonConformalCalibrator
from shuwei_c.losses import joint_forecast_loss
from shuwei_c.models import build_model


def main():
    data = {"input_dim": 10, "horizon": 12, "target_dim": 3}
    config = {"name": "quality_gru", "hidden_dim": 24, "num_layers": 1, "dropout": 0.0, "value_dim": 6, "evidence_indices": [6, 7, 8, 9]}
    x = torch.randn(16, 15, 10); x[:, :, 6] = torch.randint(0, 2, (16, 15)); x[:, :, 8] = torch.randint(0, 2, (16, 15))
    target = torch.randn(16, 12, 3) * 0.1; mask = torch.ones(16, 12, dtype=torch.bool); reliability = x[:, :, 8].clone()
    output = build_model(config, data)(x)
    loss, parts = joint_forecast_loss(output, target, mask, reliability, 0.2, 0.02); loss.backward()
    calibrator = HorizonConformalCalibrator(0.9).fit(output.mean.detach(), output.variance.detach(), target, mask)
    radius = calibrator.radius(output.variance.detach())
    assert output.reliability_logits.shape == reliability.shape and radius.shape == mask.shape
    assert torch.isfinite(loss) and torch.all(radius >= 0)
    print("joint quality-forecast model OK", {k: float(v) for k, v in parts.items()})
    print("horizon conformal calibration OK", radius.shape)


if __name__ == "__main__": main()
