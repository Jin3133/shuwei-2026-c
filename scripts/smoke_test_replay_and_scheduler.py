#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.active.replay import MissingObservationReplay
from shuwei_c.scheduling import Execute, Job, Review, ScenarioConfig, ScenarioSimulator, Stop, Wait


def test_replay():
    t = np.arange(0, 1.1, 0.1); states = np.stack([t, t * 2, t * 0], axis=1)
    replay = MissingObservationReplay(t, states, np.ones(len(t), bool), 0.3, 0.8, 0.2, 2)
    while replay.now_s < 0.4: replay.advance()
    assert replay.visible_observation() is None
    replay.request(); assert replay.advance() is None
    returned = replay.advance(); assert returned is not None and returned.timestamp_s == 0.4


def test_scheduler():
    jobs = [Job("J1", 0, 5, 2, "Z1"), Job("J2", 0, 10, 1, "Z2", ("J1",))]
    cfg = ScenarioConfig(20, 20, 2, {"Z0": (0, 0), "Z1": (1, 0), "Z2": (0, 1)})
    sim = ScenarioSimulator(jobs, {"J1": 2.0, "J2": 3.0}, cfg)
    sim.step(Review("J1")); sim.step(Execute("J1", 2.5)); sim.step(Execute("J2", 2.5)); sim.step(Stop())
    metrics = sim.metrics(); assert metrics["valid_protocol"] and metrics["weighted_completion_rate"] == 1.0
    assert abs(metrics["reserved_idle_s"] - 0.5) < 1e-9 and abs(metrics["observed_overrun_s"] - 0.5) < 1e-9

    invalid = ScenarioSimulator(jobs, {"J1": 2.0, "J2": 3.0}, cfg)
    invalid.step(Execute("J2", 2.0)); assert not invalid.metrics()["valid_protocol"]


if __name__ == "__main__":
    test_replay(); test_scheduler(); print("Q2 causal replay OK"); print("Q3 official-rule simulator OK")
