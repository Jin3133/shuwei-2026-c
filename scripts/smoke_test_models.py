#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.datasets import make_synthetic_windows
from shuwei_c.losses import forecast_loss
from shuwei_c.metrics import trajectory_metrics
from shuwei_c.models import build_model


def main():
    data_cfg = {"input_dim": 10, "horizon": 12, "target_dim": 3}
    x, y, mask = make_synthetic_windows(8, 15, 12, 10)
    x, y, mask = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)
    configs = [
        {"name": "last_observation"},
        {"name": "constant_velocity", "sample_period_s": 1 / 30, "velocity_indices": [3, 4, 5]},
        {"name": "gru", "hidden_dim": 24, "num_layers": 1, "dropout": 0.0, "probabilistic": True},
        {"name": "tcn", "channels": [16, 24], "kernel_size": 3, "dropout": 0.0, "probabilistic": True},
    ]
    for config in configs:
        output = build_model(config, data_cfg)(x)
        assert output.mean.shape == y.shape
        loss_name = "gaussian_nll" if output.log_variance is not None else "mse"
        loss = forecast_loss(output, y, mask, loss_name)
        assert torch.isfinite(loss)
        print(config["name"], f"loss={loss.item():.6f}", trajectory_metrics(output, y, mask))


if __name__ == "__main__": main()

