#!/usr/bin/env python3
"""Train-only environment-grouped OOF calibration for Q3 duration quantiles."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config


def quantile_higher(values, q): return float(np.quantile(values, q, method="higher"))


def pinball(y, pred, q):
    error = y - pred
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = load_config(args.experiment); rows = pq.read_table(cfg["source"]).to_pylist()
    with Path(cfg["official_records"]).open(encoding="utf-8-sig", newline="") as stream:
        environment = {r["episode_uuid"]: r["environment_id"] for r in csv.DictReader(stream)}
    excluded = set(cfg["features"]["excluded"]) | set(cfg["features"]["categorical"])
    numeric = [name for name in rows[0] if name not in excluded]
    families = sorted({r["family_id"] for r in rows})
    train = [r for r in rows if r["split"] == cfg["train_split"]]; valid = [r for r in rows if r["split"] == cfg["evaluation_split"]]
    medians = {name: float(np.nanmedian([r[name] for r in train if r[name] is not None])) for name in numeric}

    def matrix(part):
        return np.asarray([[medians[name] if r[name] is None else r[name] for name in numeric] + [float(r["family_id"] == family) for family in families] for r in part], float)

    x_train, x_valid = matrix(train), matrix(valid)
    y_train = np.asarray([r["target_duration_s"] for r in train], float); y_valid = np.asarray([r["target_duration_s"] for r in valid], float)
    groups = np.asarray([environment[r["episode_uuid"]] for r in train]); prefixes_train = np.asarray([r["prefix_duration_s"] for r in train], float)
    prefixes_valid = np.asarray([r["prefix_duration_s"] for r in valid], float); levels = list(map(float, cfg["quantiles"])); params = cfg["model"]
    splitter = GroupKFold(n_splits=cfg["n_group_folds"]); oof = np.zeros((len(train), len(levels))); final_raw = np.zeros((len(valid), len(levels)))
    for qi, q in enumerate(levels):
        for fit_ids, held_ids in splitter.split(x_train, y_train, groups):
            model = GradientBoostingRegressor(loss="quantile", alpha=q, **params); model.fit(x_train[fit_ids], y_train[fit_ids]); oof[held_ids, qi] = model.predict(x_train[held_ids])
        final = GradientBoostingRegressor(loss="quantile", alpha=q, **params); final.fit(x_train, y_train); final_raw[:, qi] = final.predict(x_valid)
    corrections = {}
    calibrated = final_raw.copy()
    for prefix in cfg["prefixes_s"]:
        train_ids = np.isclose(prefixes_train, prefix); valid_ids = np.isclose(prefixes_valid, prefix)
        corrections[str(prefix)] = {}
        for qi, q in enumerate(levels):
            correction = quantile_higher(y_train[train_ids] - oof[train_ids, qi], q)
            corrections[str(prefix)][str(q)] = correction; calibrated[valid_ids, qi] += correction
    raw_crossing = float(np.mean(np.any(np.diff(final_raw, axis=1) < 0, axis=1)))
    calibrated_crossing = float(np.mean(np.any(np.diff(calibrated, axis=1) < 0, axis=1)))
    calibrated = np.maximum.accumulate(calibrated, axis=1)
    summaries = []
    for label, pred in (("uncalibrated", np.maximum.accumulate(final_raw, axis=1)), ("oof_calibrated", calibrated)):
        for prefix in cfg["prefixes_s"]:
            ids = np.isclose(prefixes_valid, prefix); truth = y_valid[ids]
            row = {"variant": label, "prefix_s": prefix, "n": int(ids.sum()), "q50_mae_s": float(np.mean(np.abs(truth - pred[ids, 0]))), "q50_rmse_s": float(np.sqrt(np.mean((truth - pred[ids, 0])**2)))}
            for qi, q in enumerate(levels):
                reserve = pred[ids, qi]
                row[f"q{int(q*100)}_pinball"] = pinball(truth, reserve, q); row[f"q{int(q*100)}_coverage"] = float(np.mean(truth <= reserve))
                row[f"q{int(q*100)}_mean_overrun_s"] = float(np.mean(np.maximum(truth - reserve, 0))); row[f"q{int(q*100)}_mean_idle_s"] = float(np.mean(np.maximum(reserve - truth, 0)))
            summaries.append(row)
    report = {"experiment_id": cfg["experiment_id"], "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "train_environments": len(set(groups)), "folds": cfg["n_group_folds"], "corrections_s": corrections, "raw_crossing_rate": raw_crossing, "pre_monotone_calibrated_crossing_rate": calibrated_crossing, "results": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
