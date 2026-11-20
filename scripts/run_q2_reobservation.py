#!/usr/bin/env python3
"""Replay fixed-interval and uncertainty-triggered re-observation policies."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config
from shuwei_c.models import build_model


def revision(): return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def interpolate(elapsed: float, horizons, values):
    if elapsed <= horizons[0]: return values[0] * elapsed / horizons[0]
    for left, right, a, b in zip(horizons[:-1], horizons[1:], values[:-1], values[1:]):
        if elapsed <= right:
            weight = (elapsed - left) / (right - left)
            return a * (1 - weight) + b * weight
    return values[-1]


def radius_at(age: float, horizons, radii):
    if age <= horizons[0]: return radii[0] * age / horizons[0]
    if age <= horizons[-1]: return float(interpolate(age, horizons, radii))
    slope = (radii[-1] - radii[-2]) / (horizons[-1] - horizons[-2])
    return radii[-1] + slope * (age - horizons[-1])


def replay(row, initial_displacements, horizons, radii, policy, cfg):
    times = np.asarray(row["target_timestamps_s"], float)
    truth = np.asarray(row["target_position"], float)
    start = float(row["mask_start_s"]); last_visible = np.asarray(row["last_visible_position"], float)
    observations = [(start, last_visible.copy())]
    pending = None; requests = 0; errors = []; next_fixed = start + cfg["fixed_interval_s"]
    for now, reference in zip(times, truth):
        if pending is not None and now + 1e-12 >= pending[0]:
            observations.append((pending[1], pending[2])); pending = None
        last_time, last_position = observations[-1]
        age = max(now - last_time, 0.0)
        if len(observations) == 1:
            prediction = last_visible + interpolate(now - start, horizons, initial_displacements)
        else:
            previous_time, previous_position = observations[-2]
            velocity = (last_position - previous_position) / max(last_time - previous_time, 1e-6)
            prediction = last_position + math.exp(-cfg["shrinkage_decay"] * age) * velocity * age
        should_request = False
        if pending is None and requests < cfg["request_budget"]:
            if policy[0] == "fixed" and now + 1e-12 >= next_fixed:
                should_request = True; next_fixed += cfg["fixed_interval_s"]
            elif policy[0] == "uncertainty" and radius_at(age, horizons, radii) > policy[1]:
                should_request = True
        if should_request:
            pending = (now + cfg["return_delay_s"], now, reference.copy()); requests += 1
        errors.append(float(np.linalg.norm(prediction - reference)))
    return {"path_error": float(np.mean(errors)), "endpoint_error": errors[-1], "requests": requests, "cost": requests * cfg["request_cost"], "wait_exposure": requests * cfg["return_delay_s"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = load_config(args.experiment); model_cfg = load_config(cfg["model_config"])
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(model_cfg["model"], model_cfg["data"]); model.load_state_dict(checkpoint["model"]); model.eval()
    norm = json.loads(Path(cfg["normalization"]).read_text()); calibration = json.loads(Path(cfg["calibration_report"]).read_text())
    pm, ps = np.asarray(norm["position"]["mean"], np.float32), np.asarray(norm["position"]["scale"], np.float32)
    vm, vs = np.asarray(norm["velocity"]["mean"], np.float32), np.asarray(norm["velocity"]["scale"], np.float32)
    horizons = np.asarray(model_cfg["data"]["target_horizons_s"], float)
    radii = np.asarray([r["mean_calibrated_radius_m"] for r in calibration["per_horizon"]], float)
    table = pq.read_table(cfg["private_scoring_table"]); table = table.filter(pc.equal(table["split"], cfg["split"])); rows = table.to_pylist()
    policies = [("none", None), ("fixed", cfg["fixed_interval_s"])] + [("uncertainty", float(x)) for x in cfg["uncertainty_thresholds_m"]]
    collected = defaultdict(list)
    for start_idx in range(0, len(rows), 512):
        batch = rows[start_idx:start_idx + 512]
        pos = np.asarray([r["history_position"] for r in batch], np.float32); vel = np.asarray([r["history_velocity"] for r in batch], np.float32)
        quality = np.asarray([r["history_quality"] for r in batch], np.float32)[:, :, 0]; hand = np.asarray([r["hand"] == "right" for r in batch], np.float32)
        x = np.zeros((len(batch), pos.shape[1], 10), np.float32); x[:, :, :3] = (pos - pm) / ps; x[:, :, 3:6] = (vel - vm) / vs
        x[:, :, 6] = 1; x[:, :, 8] = quality; x[:, :, 9] = hand[:, None]
        with torch.no_grad(): means = model(torch.from_numpy(x)).mean.numpy()
        shrink = np.exp(-cfg["shrinkage_decay"] * horizons)
        for i, row in enumerate(batch):
            initial = means[i] * shrink[:, None]
            duration = float(row["mask_duration_s"])
            for policy in policies:
                result = replay(row, initial, horizons, radii, policy, cfg)
                name = "no_request" if policy[0] == "none" else "fixed_interval" if policy[0] == "fixed" else f"uncertainty_{policy[1]:.2f}"
                collected[duration, name].append(result)
    summaries = []
    for (duration, name), values in sorted(collected.items()):
        summaries.append({"mask_duration_s": duration, "policy": name, "n": len(values), **{f"mean_{key}": float(np.mean([v[key] for v in values])) for key in values[0]}})
    aggregate = []
    names = sorted({name for _, name in collected})
    for name in names:
        values = [v for (duration, policy), group in collected.items() if policy == name for v in group]
        path_error = float(np.mean([v["path_error"] for v in values])); requests = float(np.mean([v["requests"] for v in values]))
        aggregate.append({"policy": name, "n": len(values), "mean_path_error_m": path_error, "mean_endpoint_error_m": float(np.mean([v["endpoint_error"] for v in values])), "mean_requests": requests, "mean_cost": float(np.mean([v["cost"] for v in values])), "mean_wait_exposure_s": float(np.mean([v["wait_exposure"] for v in values])), "selection_objective": path_error + 0.02 * requests})
    report = {"experiment_id": cfg["experiment_id"], "git_revision": revision(), "conditions": {k: cfg[k] for k in ("split", "return_delay_s", "request_cost", "request_budget", "fixed_interval_s", "selection_objective")}, "aggregate": aggregate, "by_duration": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
