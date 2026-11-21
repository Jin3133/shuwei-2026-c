from __future__ import annotations

from scripts.run_q3_nominal_scheduling import choose_action
from shuwei_c.scheduling.simulator import Execute, Job, ScenarioConfig, ScenarioSimulator, Wait


def make_simulator():
    jobs = [
        Job("J1", 0.0, 15.0, 1.0, "Z1"),
        Job("J2", 0.0, 18.0, 2.0, "Z1"),
        Job("J3", 20.0, 40.0, 100.0, "Z1"),
    ]
    config = ScenarioConfig(
        horizon_s=60.0,
        cost_budget=100.0,
        review_limit=0,
        zones={"Z0": (0.0, 0.0), "Z1": (1.0, 0.0)},
    )
    return jobs, ScenarioSimulator(jobs, {"J1": 2.0, "J2": 2.0, "J3": 2.0}, config)


def test_unreleased_estimate_cannot_change_current_action():
    jobs, sim = make_simulator()
    visible = {"J1": {"q50": 2.0, "q80": 3.0}, "J2": {"q50": 4.0, "q80": 5.0}}
    ordinary_future = {**visible, "J3": {"q50": 3.0, "q80": 4.0}}
    adversarial_future = {**visible, "J3": {"q50": 1_000_000.0, "q80": 1_000_001.0}}

    a = choose_action(sim, jobs, ordinary_future, "apparent_tardiness_cost", 0.0, 2.0)
    b = choose_action(sim, jobs, adversarial_future, "apparent_tardiness_cost", 0.0, 2.0)

    assert isinstance(a, Execute)
    assert a == b


def test_future_estimates_are_not_required_before_release():
    jobs, sim = make_simulator()
    visible_only = {"J1": {"q50": 2.0, "q80": 3.0}, "J2": {"q50": 4.0, "q80": 5.0}}
    action = choose_action(sim, jobs, visible_only, "apparent_tardiness_cost", 0.0, 2.0)
    assert isinstance(action, Execute)


def test_no_ready_job_waits_without_future_estimate():
    jobs, sim = make_simulator()
    sim.step(Execute("J1", 3.0)); sim.step(Execute("J2", 5.0))
    action = choose_action(sim, jobs, {}, "apparent_tardiness_cost", 0.0, 2.0)
    assert isinstance(action, Wait)
    assert action.duration_s == 10.0
