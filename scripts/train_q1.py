#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config
from shuwei_c.datasets import TrajectoryWindowDataset
from shuwei_c.engine import fit
from shuwei_c.models import build_model


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); args = p.parse_args(); cfg = load_config(args.config)
    seed = cfg["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device_name = cfg["device"]; device = torch.device("mps" if device_name == "auto" and torch.backends.mps.is_available() else "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    data = cfg["data"]; train_ds = TrajectoryWindowDataset(data["train_path"]); val_ds = TrajectoryWindowDataset(data["val_path"])
    train_loader = DataLoader(train_ds, batch_size=data["batch_size"], shuffle=True, num_workers=data["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=data["batch_size"], shuffle=False, num_workers=data["num_workers"])
    model = build_model(cfg["model"], data).to(device); history = fit(model, train_loader, val_loader, cfg, device)
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__": main()

