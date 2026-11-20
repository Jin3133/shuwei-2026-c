#!/usr/bin/env python3
"""Evaluate paid-review policies on the frozen Q3 scheduling policy."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from shuwei_c.config import load_config
from shuwei_c.scheduling.simulator import Execute, Job, Review, ScenarioConfig, ScenarioSimulator
from run_q3_nominal_scheduling import aggregate, choose_action, read_csv


def build_predictions(rows, cfg, corrections):
    if cfg.get("model_kind") == "random_forest":
        return build_random_forest_predictions(rows, cfg, corrections)
    train = [r for r in rows if r["split"] == cfg["train_split"]]
    evaluation = [r for r in rows if r["split"] == cfg["evaluation_split"]]
    excluded = {"prefix_id", "segment_id", "episode_uuid", "dataset", "split", "family_id", "target_duration_s"}
    numeric = [name for name in rows[0] if name not in excluded]
    families = sorted({r["family_id"] for r in rows})
    medians = {name: float(np.nanmedian([r[name] for r in train if r[name] is not None])) for name in numeric}

    def matrix(part):
        return np.asarray([
            [medians[n] if r[n] is None else r[n] for n in numeric]
            + [float(r["family_id"] == family) for family in families]
            for r in part
        ], float)

    x_train, x_eval = matrix(train), matrix(evaluation)
    y_train = np.asarray([r["target_duration_s"] for r in train], float)
    raw = {}
    for q in map(float, cfg["quantiles"]):
        model = GradientBoostingRegressor(loss="quantile", alpha=q, **cfg["model"])
        model.fit(x_train, y_train); raw[q] = model.predict(x_eval)
    result = defaultdict(dict)
    for i, row in enumerate(evaluation):
        prefix = float(row["prefix_duration_s"]); correction = float(corrections[str(prefix)]["0.8"])
        q50 = max(float(raw[0.5][i]), 0.2); q80 = max(q50, float(raw[0.8][i]) + correction)
        result[row["segment_id"]][prefix] = {"q50": q50, "q80": q80}
    return result


def build_random_forest_predictions(rows, cfg, corrections):
    excluded = {"prefix_id", "segment_id", "episode_uuid", "dataset", "split", "family_id", "target_duration_s", "prefix_duration_s"}
    numeric = [name for name in rows[0] if name not in excluded]
    families = sorted({r["family_id"] for r in rows}); result = defaultdict(dict)
    for prefix in map(float, cfg["prefixes_s"]):
        train = [r for r in rows if r["split"] == cfg["train_split"] and np.isclose(r["prefix_duration_s"], prefix)]
        evaluation = [r for r in rows if r["split"] == cfg["evaluation_split"] and np.isclose(r["prefix_duration_s"], prefix)]
        medians = {name: float(np.nanmedian([r[name] for r in train if r[name] is not None])) for name in numeric}
        def matrix(part):
            return np.asarray([[medians[n] if r[n] is None else r[n] for n in numeric] + [float(r["family_id"] == f) for f in families] for r in part], float)
        model = RandomForestRegressor(**cfg["model"]).fit(matrix(train), np.asarray([r["target_duration_s"] for r in train], float))
        tree_values = np.asarray([tree.predict(matrix(evaluation)) for tree in model.estimators_]).T
        correction = corrections[str(prefix)]
        q50 = np.quantile(tree_values, 0.5, axis=1) + float(correction[0]); q80 = np.quantile(tree_values, 0.8, axis=1) + float(correction[1])
        for i, row in enumerate(evaluation):
            result[row["segment_id"]][prefix] = {"q50": max(float(q50[i]), 0.2), "q80": max(float(q50[i]), float(q80[i]), 0.2)}
    return result


def can_review(sim, job_id):
    s, c, runtime = sim.state, sim.config, sim.state.jobs[job_id]
    return (runtime.reviews < c.max_reviews_per_job and s.review_count < c.review_limit
            and s.cost + c.review_cost <= c.cost_budget and s.now_s + c.review_delay_s <= c.horizon_s)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--trials-csv", type=Path, required=True)
    args = parser.parse_args(); cfg = load_config(args.experiment); task_root = Path(cfg["task_root"])
    common = json.loads((task_root / "common_conditions.json").read_text())
    tasks, units = read_csv(task_root / "tasks.csv"), read_csv(task_root / "source_segment_index.csv")
    zones = {r["zone_id"]: (float(r["x"]), float(r["y"])) for r in read_csv(task_root / "service_zones.csv")}
    calibration = json.loads(Path(cfg["calibration_results"]).read_text())
    if cfg.get("model_kind") == "random_forest":
        corrections = {str(r["prefix_s"]): r["corrections_s"] for r in calibration["results"] if r["model"] == "random_forest"}
    else:
        corrections = calibration["corrections_s"]
    predictions = build_predictions(pq.read_table(cfg["features"]).to_pylist(), cfg, corrections)
    pools = defaultdict(list)
    for row in units:
        if row["split"] == cfg["evaluation_split"] and row["eligible"] == "1" and row["unit_id"] in predictions:
            pools[row["family_id"]].append(row)
    cases = {r["case_id"]: r for r in common["cases"]}; trials = []
    for case_id in sorted(cases):
        task_rows = [r for r in tasks if r["case_id"] == case_id]; case = cases[case_id]
        jobs = [Job(r["job_id"], float(r["release_s"]), float(r["due_s"]), float(r["priority_weight"]), r["zone_id"],
                    tuple(x.strip() for x in r["predecessors"].split(";") if x.strip())) for r in task_rows]
        scenario = ScenarioConfig(
            float(case["horizon_s"]), float(case["cost_budget"]), int(case["observation_limit"]), zones,
            initial_zone=common["initial_zone"], review_delay_s=float(common["observation"]["delay_s"]),
            review_cost=float(common["observation"]["cost"]), max_reviews_per_job=int(common["observation"]["max_per_job"]),
            reserve_min_s=float(common["service"]["reserve_min_s"]), changeover_s=float(common["movement"]["changeover_s"]),
            service_fixed_cost=float(common["fees"]["service_fixed"]), reserve_cost_rate=float(common["fees"]["reserve_per_second"]),
            distance_cost_rate=float(common["fees"]["distance_per_unit"]), max_actions=int(common["max_actions"]))
        for seed in common["validation_seeds"]:
            rng = random.Random(int(seed)); sampled = {r["job_id"]: rng.choice(pools[r["family_id"]]) for r in task_rows}
            realized = {j: float(r["duration_s"]) * float(cfg["scenario"]["alpha"]) + float(cfg["scenario"]["beta_s"]) for j, r in sampled.items()}
            for policy in cfg["policies"]:
                sim = ScenarioSimulator(jobs, realized, scenario)
                while not sim.state.stopped:
                    estimates = {}
                    for job_id, unit in sampled.items():
                        prefix = cfg["prefixes_s"][sim.state.jobs[job_id].reviews]
                        pred = predictions[unit["unit_id"]][float(prefix)]; fraction = float(cfg["reservation_fraction"])
                        estimates[job_id] = {"q50": pred["q50"], "q80": pred["q50"] + fraction * (pred["q80"] - pred["q50"]), "spread": pred["q80"] - pred["q50"]}
                    action = choose_action(sim, jobs, estimates, cfg["ordering"]["rule"], float(cfg["scenario"]["beta_s"]), float(cfg["ordering"]["lookahead_factor"]))
                    if isinstance(action, Execute) and can_review(sim, action.job_id):
                        runtime = sim.state.jobs[action.job_id]
                        should_review = policy["mode"] == "fixed_one" and runtime.reviews == 0
                        if policy["mode"] == "uncertainty":
                            should_review = estimates[action.job_id]["spread"] > float(policy["spread_threshold_s"])
                        if should_review: action = Review(action.job_id)
                    sim.step(action)
                trials.append({"case_id": case_id, "seed": int(seed), "policy": policy["name"], **sim.metrics(),
                               "sampled_units": ";".join(sampled[j.job_id]["unit_id"] for j in jobs)})
    args.trials_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.trials_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trials[0])); writer.writeheader(); writer.writerows(trials)
    summaries = []
    for case_id in sorted(cases):
        for policy in cfg["policies"]:
            part = [r for r in trials if r["case_id"] == case_id and r["policy"] == policy["name"]]
            summaries.append({"case_id": case_id, "policy": policy["name"], **aggregate(part)})
    report = {"experiment_id": cfg["experiment_id"], "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "paired_seeds": len(common["validation_seeds"]), "reservation_fraction": cfg["reservation_fraction"], "summaries": summaries}
    args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
