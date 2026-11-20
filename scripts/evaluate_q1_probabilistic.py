#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.calibration import HorizonConformalCalibrator
from shuwei_c.config import load_config
from shuwei_c.datasets import TrajectoryWindowDataset
from shuwei_c.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg["model"], cfg["data"]); model.load_state_dict(checkpoint["model"]); model.eval()
    loader = DataLoader(TrajectoryWindowDataset(cfg["data"]["val_path"]), batch_size=1024, shuffle=False)
    means, variances, targets, masks = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["x"])
            means.append(output.mean); variances.append(output.variance); targets.append(batch["y"]); masks.append(batch["target_mask"])
    mean, variance, target, mask = map(torch.cat, (means, variances, targets, masks))
    calibrator = HorizonConformalCalibrator(cfg["calibration"]["coverage"]).fit(mean, variance, target, mask)
    calibrated_radius = calibrator.radius(variance)
    distance = torch.linalg.vector_norm(mean - target, dim=-1)
    sigma = variance.sqrt(); raw_joint_95 = ((target >= mean - 1.96 * sigma) & (target <= mean + 1.96 * sigma)).all(dim=-1)
    rows = []
    for i, horizon in enumerate(cfg["data"]["target_horizons_s"]):
        valid = mask[:, i]
        d = distance[:, i][valid]
        rows.append({
            "horizon_s": horizon,
            "n": int(valid.sum()),
            "mean_error_m": d.mean().item(),
            "rmse_m": torch.sqrt((d**2).mean()).item(),
            "mean_axis_std_m": sigma[:, i][valid].mean().item(),
            "raw_joint_95_coverage": raw_joint_95[:, i][valid].float().mean().item(),
            "conformal_target_coverage": cfg["calibration"]["coverage"],
            "conformal_scale": calibrator.scale[i].item(),
            "in_sample_calibrated_coverage": (d <= calibrated_radius[:, i][valid]).float().mean().item(),
            "mean_calibrated_radius_m": calibrated_radius[:, i][valid].mean().item(),
        })
    report = {
        "checkpoint_epoch": checkpoint["metrics"]["epoch"],
        "checkpoint_validation_nll": checkpoint["metrics"]["loss"],
        "calibration_split": "validation",
        "calibration_note": "Coverage shown after conformal fitting is in-sample diagnostic; frozen test coverage is the final estimate.",
        "per_horizon": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
