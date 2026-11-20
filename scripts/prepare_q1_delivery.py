#!/usr/bin/env python3
"""Convert the delivered nested Q1 parquet into the project's tensor-window NPZ contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def convert(source: Path, normalization: Path, output: Path, splits: list[str]):
    stats = json.loads(normalization.read_text(encoding="utf-8"))
    position_mean = np.asarray(stats["position"]["mean"], dtype=np.float32)
    position_scale = np.asarray(stats["position"]["scale"], dtype=np.float32)
    velocity_mean = np.asarray(stats["velocity"]["mean"], dtype=np.float32)
    velocity_scale = np.asarray(stats["velocity"]["scale"], dtype=np.float32)
    table = pq.read_table(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source), "feature_order": ["dx", "dy", "dz", "vx", "vy", "vz", "observed_mask", "time_since_valid_s", "raw_quality", "hand_side"], "target_horizons_s": [0.2, 0.5, 1.0], "splits": {}}
    for split in splits:
        part = table.filter(pc.equal(table["split"], split))
        position = np.asarray(part["input_relative_position"].to_pylist(), dtype=np.float32)
        velocity = np.asarray(part["input_velocity"].to_pylist(), dtype=np.float32)
        quality = np.asarray(part["input_quality"].to_pylist(), dtype=np.float32)
        target = np.asarray(part["target_displacement"].to_pylist(), dtype=np.float32)
        target_mask = np.asarray(part["target_ref_ok"].to_pylist(), dtype=bool)
        hand = np.asarray([0.0 if h == "left" else 1.0 for h in part["hand"].to_pylist()], dtype=np.float32)
        x = np.zeros((len(part), position.shape[1], 10), dtype=np.float32)
        x[:, :, :3] = (position - position_mean) / position_scale
        x[:, :, 3:6] = (velocity - velocity_mean) / velocity_scale
        x[:, :, 6] = 1.0
        x[:, :, 7] = 0.0
        x[:, :, 8] = quality[:, :, 0]
        x[:, :, 9] = hand[:, None]
        path = output / f"q1_{split}.npz"
        np.savez_compressed(path, x=x, y=target, target_mask=target_mask)
        manifest["splits"][split] = {"rows": len(part), "path": str(path), "x_shape": list(x.shape), "y_shape": list(target.shape)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args(); convert(args.source, args.normalization, args.output, args.splits)
