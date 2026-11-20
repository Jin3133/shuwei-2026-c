#!/usr/bin/env python3
"""Evaluate Q2 direct endpoint prediction without exposing masked targets to models."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config
from shuwei_c.models import build_model


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); exp = load_config(args.experiment)
    model_cfg = load_config(exp["model_config"])
    checkpoint = torch.load(exp["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(model_cfg["model"], model_cfg["data"]); model.load_state_dict(checkpoint["model"]); model.eval()
    norm = json.loads(Path(exp["normalization"]).read_text(encoding="utf-8"))
    calibration = json.loads(Path(exp["calibration_report"]).read_text(encoding="utf-8"))
    scale = {float(r["horizon_s"]): float(r["conformal_scale"]) for r in calibration["per_horizon"]}
    horizon_index = {float(h): i for i, h in enumerate(model_cfg["data"]["target_horizons_s"])}
    position_mean = np.asarray(norm["position"]["mean"], np.float32); position_scale = np.asarray(norm["position"]["scale"], np.float32)
    velocity_mean = np.asarray(norm["velocity"]["mean"], np.float32); velocity_scale = np.asarray(norm["velocity"]["scale"], np.float32)
    table = pq.read_table(exp["private_scoring_table"])
    table = table.filter(pc.equal(table["split"], exp["split"]))
    allowed = set(map(float, exp["mask_durations_s"]))
    table = table.filter(pc.is_in(table["mask_duration_s"], value_set=pa.array(sorted(allowed), type=pa.float64())))
    rows = table.to_pylist(); grouped = defaultdict(list)
    batch_size = 512
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        position = np.asarray([r["history_position"] for r in batch_rows], np.float32)
        velocity = np.asarray([r["history_velocity"] for r in batch_rows], np.float32)
        quality = np.asarray([r["history_quality"] for r in batch_rows], np.float32)[:, :, 0]
        hand = np.asarray([0.0 if r["hand"] == "left" else 1.0 for r in batch_rows], np.float32)
        x = np.zeros((len(batch_rows), position.shape[1], 10), np.float32)
        x[:, :, :3] = (position - position_mean) / position_scale
        x[:, :, 3:6] = (velocity - velocity_mean) / velocity_scale
        x[:, :, 6] = 1.0; x[:, :, 8] = quality; x[:, :, 9] = hand[:, None]
        with torch.no_grad(): output = model(torch.from_numpy(x))
        for i, row in enumerate(batch_rows):
            duration = float(row["mask_duration_s"]); hi = horizon_index[duration]
            truth = np.asarray(row["target_position"][-1], float)
            last = np.asarray(row["last_visible_position"], float)
            predictions = {
                "last_position": last,
                "constant_velocity": last + duration * velocity[i, -1],
                "gaussian_gru": last + output.mean[i, hi].numpy(),
            }
            for method, prediction in predictions.items(): grouped[duration, method].append(float(np.linalg.norm(prediction - truth)))
            grouped[duration, "truth_displacement"].append(truth - last)
            grouped[duration, "gru_displacement"].append(output.mean[i, hi].numpy())
            raw_radius = float(torch.linalg.vector_norm(output.variance[i, hi].sqrt()).item())
            grouped[duration, "gaussian_covered"].append(float(np.linalg.norm(predictions["gaussian_gru"] - truth) <= raw_radius * scale[duration]))
            grouped[duration, "gaussian_radius"].append(raw_radius * scale[duration])
    results = []
    for duration in sorted(allowed):
        for method in exp["methods"]:
            values = np.asarray(grouped[duration, method])
            results.append({"mask_duration_s": duration, "method": method, "n": len(values), "mean_error_m": float(values.mean()), "rmse_m": float(np.sqrt(np.mean(values**2))), "p90_error_m": float(np.quantile(values, .9))})
        results[-1].update(calibrated_coverage=float(np.mean(grouped[duration, "gaussian_covered"])), mean_calibrated_radius_m=float(np.mean(grouped[duration, "gaussian_radius"])))
        if exp.get("tune_shrinkage", False):
            truth_d = np.asarray(grouped[duration, "truth_displacement"])
            pred_d = np.asarray(grouped[duration, "gru_displacement"])
            alpha = float(np.clip(np.sum(pred_d * truth_d) / max(np.sum(pred_d * pred_d), 1e-12), 0.0, 1.0))
            errors = np.linalg.norm(alpha * pred_d - truth_d, axis=1)
            results.append({"mask_duration_s": duration, "method": "shrunk_gaussian_gru", "n": len(errors), "shrinkage_alpha": alpha, "selection_note": "alpha fitted on validation; evaluate only after freezing on test", "mean_error_m": float(errors.mean()), "rmse_m": float(np.sqrt(np.mean(errors**2))), "p90_error_m": float(np.quantile(errors, .9))})
    report = {"experiment_id": exp["experiment_id"], "git_revision": git_revision(), "split": exp["split"], "checkpoint_epoch": checkpoint["metrics"]["epoch"], "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
