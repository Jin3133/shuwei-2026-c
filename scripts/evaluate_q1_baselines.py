#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config
from shuwei_c.datasets import TrajectoryWindowDataset
from shuwei_c.engine import evaluate
from shuwei_c.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("work/results/q1_baselines.json"))
    args = parser.parse_args(); config = load_config(args.config); data = config["data"]
    loader = DataLoader(TrajectoryWindowDataset(data["val_path"]), batch_size=data["batch_size"], shuffle=False, num_workers=data["num_workers"])
    model_group = config["model"]; results = {}
    for name in model_group["names"]:
        model_config = {"name": name}
        for key in ("sample_period_s", "velocity_indices", "quality_index", "process_variance", "measurement_variance"):
            if key in model_group: model_config[key] = model_group[key]
        model = build_model(model_config, data)
        try:
            results[name] = evaluate(model, loader, torch.device("cpu"))
        except RuntimeError as exc:
            results[name] = {"status": "unavailable", "reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
