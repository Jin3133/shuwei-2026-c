#!/usr/bin/env python3
"""Sensitivity analysis for the value of Q3 paid reviews."""
from __future__ import annotations

import argparse, csv, json, random, subprocess, sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from shuwei_c.config import load_config
from shuwei_c.scheduling.simulator import Execute, Job, Review, ScenarioConfig, ScenarioSimulator
from run_q3_nominal_scheduling import aggregate, choose_action, read_csv, visible_estimates
from run_q3_paid_review_sweep import build_predictions, can_review


def conditions(cfg):
    result = []
    for delay, cost in product(cfg["observation_grid"]["delay_s"], cfg["observation_grid"]["cost"]):
        result.append({"condition": f"obs_d{delay}_c{cost}", "delay_s": float(delay), "review_cost": float(cost), "alpha": 1.0, "beta_s": 0.5})
    for service in cfg["service_sensitivity"]:
        alpha, beta = float(service["alpha"]), float(service["beta_s"])
        result.append({"condition": f"service_a{alpha}_b{beta}", "delay_s": 1.0, "review_cost": 1.0, "alpha": alpha, "beta_s": beta})
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--trials-csv", type=Path, required=True)
    args = ap.parse_args(); cfg = load_config(args.experiment); root = Path(cfg["task_root"])
    common = json.loads((root / "common_conditions.json").read_text()); tasks = read_csv(root / "tasks.csv")
    units = read_csv(root / "source_segment_index.csv"); zones = {r["zone_id"]: (float(r["x"]), float(r["y"])) for r in read_csv(root / "service_zones.csv")}
    calibration = json.loads(Path(cfg["calibration_results"]).read_text())
    predictions = build_predictions(pq.read_table(cfg["features"]).to_pylist(), cfg, calibration["corrections_s"])
    pools = defaultdict(list)
    for r in units:
        if r["split"] == cfg["evaluation_split"] and r["eligible"] == "1" and r["unit_id"] in predictions: pools[r["family_id"]].append(r)
    cases = {r["case_id"]: r for r in common["cases"]}; trials = []; conds = conditions(cfg)
    for case_id in sorted(cases):
        task_rows = [r for r in tasks if r["case_id"] == case_id]; case = cases[case_id]
        jobs = [Job(r["job_id"], float(r["release_s"]), float(r["due_s"]), float(r["priority_weight"]), r["zone_id"],
                    tuple(x.strip() for x in r["predecessors"].split(";") if x.strip())) for r in task_rows]
        for seed in common["validation_seeds"]:
            rng = random.Random(int(seed)); sampled = {r["job_id"]: rng.choice(pools[r["family_id"]]) for r in task_rows}
            for cond in conds:
                scenario = ScenarioConfig(float(case["horizon_s"]), float(case["cost_budget"]), int(case["observation_limit"]), zones,
                    initial_zone=common["initial_zone"], review_delay_s=cond["delay_s"], review_cost=cond["review_cost"],
                    max_reviews_per_job=int(common["observation"]["max_per_job"]), reserve_min_s=float(common["service"]["reserve_min_s"]),
                    changeover_s=float(common["movement"]["changeover_s"]), service_fixed_cost=float(common["fees"]["service_fixed"]),
                    reserve_cost_rate=float(common["fees"]["reserve_per_second"]), distance_cost_rate=float(common["fees"]["distance_per_unit"]),
                    max_actions=int(common["max_actions"]))
                realized = {j: float(r["duration_s"]) * cond["alpha"] + cond["beta_s"] for j, r in sampled.items()}
                for policy in ("no_review", "uncertainty_review"):
                    sim = ScenarioSimulator(jobs, realized, scenario)
                    while not sim.state.stopped:
                        estimates = {}
                        for job_id, unit in sampled.items():
                            prefix = sim.visible_prefix_s(job_id, cfg["prefixes_s"])
                            if prefix is None: continue
                            pred = predictions[unit["unit_id"]][prefix]
                            q50 = cond["alpha"] * pred["q50"]; q80 = cond["alpha"] * pred["q80"]
                            fraction = float(cfg["reservation_fraction"])
                            estimates[job_id] = {"q50": q50, "q80": q50 + fraction * (q80 - q50), "spread": q80 - q50}
                        estimates = visible_estimates(sim, jobs, estimates)
                        action = choose_action(sim, jobs, estimates, cfg["ordering"]["rule"], cond["beta_s"], float(cfg["ordering"]["lookahead_factor"]))
                        if policy == "uncertainty_review" and isinstance(action, Execute) and can_review(sim, action.job_id):
                            if estimates[action.job_id]["spread"] > float(cfg["spread_threshold_s"]): action = Review(action.job_id)
                        sim.step(action)
                    trials.append({"case_id": case_id, "seed": int(seed), "condition": cond["condition"], "delay_s": cond["delay_s"],
                        "review_cost": cond["review_cost"], "alpha": cond["alpha"], "beta_s": cond["beta_s"], "policy": policy,
                        **sim.metrics(), "sampled_units": ";".join(sampled[j.job_id]["unit_id"] for j in jobs)})
    args.trials_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.trials_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0])); w.writeheader(); w.writerows(trials)
    summaries = []
    for case_id in sorted(cases):
        for cond in conds:
            for policy in ("no_review", "uncertainty_review"):
                part = [r for r in trials if r["case_id"] == case_id and r["condition"] == cond["condition"] and r["policy"] == policy]
                summaries.append({"case_id": case_id, **cond, "policy": policy, **aggregate(part)})
    report = {"experiment_id": cfg["experiment_id"], "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "paired_seeds": len(common["validation_seeds"]), "spread_threshold_s": cfg["spread_threshold_s"], "summaries": summaries}
    args.output.write_text(json.dumps(report, indent=2) + "\n"); print(f"wrote {len(trials)} trials and {len(summaries)} summaries")


if __name__ == "__main__": main()
