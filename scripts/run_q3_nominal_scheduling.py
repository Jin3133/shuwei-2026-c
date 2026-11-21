#!/usr/bin/env python3
"""Run paired nominal Q3 scheduling trials with frozen duration predictors."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config
from shuwei_c.scheduling.simulator import Execute, Job, ScenarioConfig, ScenarioSimulator, Stop, Wait


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def quantile_higher(values, q):
    return float(np.quantile(values, q, method="higher"))


def build_predictors(rows, cfg, correction):
    prefix = float(cfg["prefix_s"])
    train = [r for r in rows if r["split"] == cfg["train_split"]]
    evaluation = [r for r in rows if r["split"] == cfg["evaluation_split"] and np.isclose(r["prefix_duration_s"], prefix)]
    excluded = {"prefix_id", "segment_id", "episode_uuid", "dataset", "split", "family_id", "target_duration_s"}
    numeric = [name for name in rows[0] if name not in excluded]
    families = sorted({r["family_id"] for r in rows})
    medians = {name: float(np.nanmedian([r[name] for r in train if r[name] is not None])) for name in numeric}

    def matrix(part):
        return np.asarray([
            [medians[name] if r[name] is None else r[name] for name in numeric]
            + [float(r["family_id"] == family) for family in families]
            for r in part
        ], dtype=float)

    x_train, x_eval = matrix(train), matrix(evaluation)
    y_train = np.asarray([r["target_duration_s"] for r in train], dtype=float)
    gbdt = {}
    for q in map(float, cfg["quantiles"]):
        model = GradientBoostingRegressor(loss="quantile", alpha=q, **cfg["model"])
        model.fit(x_train, y_train)
        gbdt[q] = model.predict(x_eval)

    family_train = defaultdict(list)
    for row in train:
        if np.isclose(row["prefix_duration_s"], prefix):
            family_train[row["family_id"]].append(float(row["target_duration_s"]))
    predictions = {}
    for i, row in enumerate(evaluation):
        family = row["family_id"]
        predictions[row["segment_id"]] = {
            "family_q50": quantile_higher(family_train[family], 0.5),
            "family_q80": quantile_higher(family_train[family], 0.8),
            "gbdt_q50": max(float(gbdt[0.5][i]), 0.2),
            "gbdt_q80": max(float(gbdt[0.8][i]) + correction, 0.2),
        }
        predictions[row["segment_id"]]["gbdt_q80"] = max(
            predictions[row["segment_id"]]["gbdt_q50"], predictions[row["segment_id"]]["gbdt_q80"]
        )
    return predictions


def choose_action(sim, jobs, estimates, ordering, beta, lookahead_factor):
    state, config = sim.state, sim.config
    candidates = []
    # Only released, unfinished and precedence-feasible jobs are visible to the
    # current ordering decision. In particular, future jobs must not affect the
    # ATC normalization through their duration estimates.
    ready_jobs = [
        job for job in jobs
        if not state.jobs[job.job_id].attempted
        and job.release_s <= state.now_s
        and all(state.jobs[p].completed for p in job.predecessors)
    ]
    if ready_jobs:
        missing = [job.job_id for job in ready_jobs if job.job_id not in estimates]
        if missing: raise KeyError(f"Missing visible estimates for ready jobs: {missing}")
        mean_process = float(np.mean([
            max(estimates[job.job_id]["q50"] + beta, config.reserve_min_s)
            for job in ready_jobs
        ]))
    else:
        mean_process = config.reserve_min_s
    for job in ready_jobs:
        x0, y0 = config.zones[state.zone_id]
        x1, y1 = config.zones[job.zone_id]
        distance = math.hypot(x0 - x1, y0 - y1)
        reserve = max(estimates[job.job_id]["q80"] + beta, config.reserve_min_s)
        process = distance + config.changeover_s + max(estimates[job.job_id]["q50"] + beta, config.reserve_min_s)
        cost = config.service_fixed_cost + config.reserve_cost_rate * reserve + config.distance_cost_rate * distance
        if state.cost + cost > config.cost_budget or state.now_s + distance + config.changeover_s + reserve > config.horizon_s:
            continue
        slack = max(job.due_s - state.now_s - process, 0.0)
        if ordering == "first_come_first_served":
            sequence = int(job.job_id.rsplit("J", 1)[-1])
            score = -1000.0 * job.release_s - sequence
        elif ordering == "earliest_due_date":
            score = -job.due_s
        elif ordering == "weighted_earliest_due_date":
            score = job.weight / max(job.due_s - state.now_s, 1.0)
        else:
            score = job.weight / max(process, 1e-9) * math.exp(-slack / max(lookahead_factor * mean_process, 1e-9))
        candidates.append((score, -job.due_s, job.job_id, reserve))
    if candidates:
        _, _, job_id, reserve = max(candidates)
        return Execute(job_id, reserve)
    future_releases = [j.release_s for j in jobs if not state.jobs[j.job_id].attempted and j.release_s > state.now_s]
    if future_releases:
        duration = min(future_releases) - state.now_s
        if state.now_s + duration <= config.horizon_s:
            return Wait(duration)
    return Stop()


def aggregate(trials):
    metric_names = [
        "weighted_completion_rate", "weighted_on_time_rate", "weighted_tardiness_s", "cost", "observations",
        "elapsed_s", "observed_overrun_s", "reserved_idle_s", "censored_units", "invalid_actions",
    ]
    result = {"n_trials": len(trials)}
    for name in metric_names:
        values = np.asarray([r[name] for r in trials], dtype=float)
        result[f"mean_{name}"] = float(values.mean())
        result[f"std_{name}"] = float(values.std())
    result["valid_protocol_rate"] = float(np.mean([r["valid_protocol"] for r in trials]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials-csv", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.experiment)
    root = Path(cfg["task_root"])
    common = json.loads((root / "common_conditions.json").read_text())
    tasks = read_csv(root / "tasks.csv")
    units = read_csv(root / "source_segment_index.csv")
    zones = {r["zone_id"]: (float(r["x"]), float(r["y"])) for r in read_csv(root / "service_zones.csv")}
    rows = pq.read_table(cfg["features"]).to_pylist()
    calibration = json.loads(Path(cfg["calibration_results"]).read_text())
    correction = float(calibration["corrections_s"][str(cfg["prefix_s"])]["0.8"])
    predictions = build_predictors(rows, cfg, correction)
    cases = {r["case_id"]: r for r in common["cases"]}
    pools = defaultdict(list)
    for row in units:
        if row["split"] == cfg["evaluation_split"] and row["eligible"] == "1" and row["unit_id"] in predictions:
            pools[row["family_id"]].append(row)
    all_trials = []
    for case_id in sorted(cases):
        case_tasks = [r for r in tasks if r["case_id"] == case_id]
        case = cases[case_id]
        jobs = [Job(
            job_id=r["job_id"], release_s=float(r["release_s"]), due_s=float(r["due_s"]),
            weight=float(r["priority_weight"]), zone_id=r["zone_id"],
            predecessors=tuple(x.strip() for x in r["predecessors"].split(";") if x.strip()),
        ) for r in case_tasks]
        scenario = ScenarioConfig(
            horizon_s=float(case["horizon_s"]), cost_budget=float(case["cost_budget"]), review_limit=int(case["observation_limit"]),
            zones=zones, initial_zone=common["initial_zone"], review_delay_s=float(common["observation"]["delay_s"]),
            review_cost=float(common["observation"]["cost"]), max_reviews_per_job=int(common["observation"]["max_per_job"]),
            reserve_min_s=float(common["service"]["reserve_min_s"]), changeover_s=float(common["movement"]["changeover_s"]),
            service_fixed_cost=float(common["fees"]["service_fixed"]), reserve_cost_rate=float(common["fees"]["reserve_per_second"]),
            distance_cost_rate=float(common["fees"]["distance_per_unit"]), max_actions=int(common["max_actions"]),
        )
        for seed in common["validation_seeds"]:
            rng = random.Random(int(seed))
            sampled = {r["job_id"]: rng.choice(pools[r["family_id"]]) for r in case_tasks}
            realized = {job_id: float(row["duration_s"]) * float(cfg["scenario"]["alpha"]) + float(cfg["scenario"]["beta_s"]) for job_id, row in sampled.items()}
            for policy_spec in cfg["policies"]:
                if isinstance(policy_spec, str):
                    policy = policy_spec
                    predictor = policy_spec
                    ordering = cfg["ordering"]["rule"]
                    reserve_fraction = 1.0
                else:
                    policy = policy_spec["name"]
                    predictor = policy_spec["predictor"]
                    ordering = policy_spec.get("ordering", cfg["ordering"]["rule"])
                    reserve_fraction = float(policy_spec.get("reserve_fraction", 1.0))
                estimates = {}
                for job_id, row in sampled.items():
                    pred = predictions[row["unit_id"]]
                    if predictor == "family_q80":
                        q50, q80 = pred["family_q50"], pred["family_q80"]
                    else:
                        q50, q80 = pred["gbdt_q50"], pred["gbdt_q80"]
                    estimates[job_id] = {"q50": q50, "q80": q50 + reserve_fraction * (q80 - q50)}
                sim = ScenarioSimulator(jobs, realized, scenario)
                while not sim.state.stopped:
                    action = choose_action(sim, jobs, estimates, ordering, float(cfg["scenario"]["beta_s"]), float(cfg["ordering"]["lookahead_factor"]))
                    sim.step(action)
                result = {"case_id": case_id, "seed": int(seed), "policy": policy, **sim.metrics()}
                result["sampled_units"] = ";".join(sampled[j.job_id]["unit_id"] for j in jobs)
                all_trials.append(result)
    args.trials_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.trials_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_trials[0])); writer.writeheader(); writer.writerows(all_trials)
    summaries = []
    for case_id in sorted(cases):
        for policy_spec in cfg["policies"]:
            policy = policy_spec if isinstance(policy_spec, str) else policy_spec["name"]
            part = [r for r in all_trials if r["case_id"] == case_id and r["policy"] == policy]
            summaries.append({"case_id": case_id, "policy": policy, **aggregate(part)})
    report = {
        "experiment_id": cfg["experiment_id"],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "paired_seeds": len(common["validation_seeds"]), "prefix_s": cfg["prefix_s"], "q80_correction_s": correction,
        "reviews_enabled": False, "summaries": summaries,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
