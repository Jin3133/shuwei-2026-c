#!/usr/bin/env python3
"""Train and evaluate simple Q3 conditional-duration quantile baselines."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config


def pinball(y, pred, q):
    error = y - pred
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = load_config(args.experiment); rows = pq.read_table(cfg["source"]).to_pylist()
    excluded = set(cfg["features"]["excluded"]) | set(cfg["features"]["categorical"])
    numeric = [k for k in rows[0] if k not in excluded]
    families = sorted({r["family_id"] for r in rows})
    train = [r for r in rows if r["split"] == cfg["train_split"]]; valid = [r for r in rows if r["split"] == cfg["evaluation_split"]]
    medians = {name: float(np.nanmedian([r[name] for r in train if r[name] is not None])) for name in numeric}

    def matrix(part):
        values = [[medians[name] if r[name] is None else r[name] for name in numeric] + [float(r["family_id"] == family) for family in families] for r in part]
        return np.asarray(values, float)

    x_train, x_valid = matrix(train), matrix(valid)
    y_train = np.asarray([r["target_duration_s"] for r in train], float); y_valid = np.asarray([r["target_duration_s"] for r in valid], float)
    levels = list(map(float, cfg["quantiles"])); model_cfg = cfg["model"]
    predictions = {"family_quantile": np.zeros((len(valid), len(levels))), "gradient_boosting_quantile": np.zeros((len(valid), len(levels)))}
    family_values = {family: np.asarray([r["target_duration_s"] for r in train if r["family_id"] == family], float) for family in families}
    for qi, q in enumerate(levels):
        predictions["family_quantile"][:, qi] = [np.quantile(family_values[r["family_id"]], q) for r in valid]
        model = GradientBoostingRegressor(loss="quantile", alpha=q, **model_cfg); model.fit(x_train, y_train)
        predictions["gradient_boosting_quantile"][:, qi] = model.predict(x_valid)
    raw_crossing = float(np.mean(np.any(np.diff(predictions["gradient_boosting_quantile"], axis=1) < 0, axis=1)))
    predictions["gradient_boosting_quantile"] = np.maximum.accumulate(predictions["gradient_boosting_quantile"], axis=1)
    summaries = []
    for method, pred in predictions.items():
        for prefix in cfg["prefixes_s"]:
            ids = np.asarray([abs(r["prefix_duration_s"] - prefix) < 1e-9 for r in valid])
            row = {"method": method, "prefix_s": prefix, "n": int(ids.sum()), "q50_mae_s": float(np.mean(np.abs(y_valid[ids] - pred[ids, 0]))), "q50_rmse_s": float(np.sqrt(np.mean((y_valid[ids] - pred[ids, 0]) ** 2)))}
            for qi, q in enumerate(levels):
                reserve = pred[ids, qi]; truth = y_valid[ids]
                row[f"q{int(q*100)}_pinball"] = pinball(truth, reserve, q)
                row[f"q{int(q*100)}_coverage"] = float(np.mean(truth <= reserve))
                row[f"q{int(q*100)}_mean_overrun_s"] = float(np.mean(np.maximum(truth - reserve, 0)))
                row[f"q{int(q*100)}_mean_idle_s"] = float(np.mean(np.maximum(reserve - truth, 0)))
            summaries.append(row)
    report = {"experiment_id": cfg["experiment_id"], "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "train_rows": len(train), "validation_rows": len(valid), "numeric_features": numeric, "families": families, "raw_quantile_crossing_rate": raw_crossing, "results": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
