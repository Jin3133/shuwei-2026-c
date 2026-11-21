"""Strict Problem 3 single-equipment online simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Job:
    job_id: str
    release_s: float
    due_s: float
    weight: float
    zone_id: str
    predecessors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioConfig:
    horizon_s: float
    cost_budget: float
    review_limit: int
    zones: dict[str, tuple[float, float]]
    initial_zone: str = "Z0"
    review_delay_s: float = 1.0
    review_cost: float = 1.0
    max_reviews_per_job: int = 2
    reserve_min_s: float = 0.2
    changeover_s: float = 0.5
    service_fixed_cost: float = 2.0
    reserve_cost_rate: float = 0.1
    distance_cost_rate: float = 0.2
    max_actions: int = 200


@dataclass(frozen=True)
class Review: job_id: str


@dataclass(frozen=True)
class Execute:
    job_id: str
    reserve_s: float


@dataclass(frozen=True)
class Wait: duration_s: float


@dataclass(frozen=True)
class Stop: pass


@dataclass
class JobRuntime:
    reviews: int = 0
    attempted: bool = False
    completed: bool = False
    completion_s: float | None = None
    service_start_s: float | None = None


@dataclass
class ScenarioState:
    now_s: float
    zone_id: str
    cost: float = 0.0
    review_count: int = 0
    action_count: int = 0
    stopped: bool = False
    valid_protocol: bool = True
    invalid_actions: int = 0
    observed_overrun_s: float = 0.0
    reserved_idle_s: float = 0.0
    censored_units: int = 0
    jobs: dict[str, JobRuntime] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)


class ScenarioSimulator:
    """Keep hidden realized durations private and expose only legal state."""

    def __init__(self, jobs: list[Job], realized_durations_s: dict[str, float], config: ScenarioConfig):
        self.jobs = {j.job_id: j for j in jobs}; self._durations = dict(realized_durations_s); self.config = config
        if set(self.jobs) != set(self._durations): raise ValueError("Each job needs exactly one hidden realized duration")
        self.state = ScenarioState(0.0, config.initial_zone, jobs={job_id: JobRuntime() for job_id in self.jobs})

    def public_snapshot(self) -> dict:
        return {
            "now_s": self.state.now_s, "zone_id": self.state.zone_id, "cost": self.state.cost,
            "review_count": self.state.review_count,
            "jobs": {j: {"reviews": r.reviews, "attempted": r.attempted, "completed": r.completed, "completion_s": r.completion_s} for j, r in self.state.jobs.items()},
        }

    def visible_prefix_s(self, job_id: str, levels=(0.2, 0.6, 1.0)) -> float | None:
        """Return the legally visible prefix, or None before job release."""
        if job_id not in self.jobs: raise KeyError(job_id)
        if self.jobs[job_id].release_s > self.state.now_s: return None
        reviews = self.state.jobs[job_id].reviews
        if reviews >= len(levels): raise ValueError("Visible-prefix levels do not cover review state")
        return float(levels[reviews])

    def _invalid(self, action, reason):
        self.state.invalid_actions += 1; self.state.valid_protocol = False; self.state.stopped = True
        self.state.log.append({"time_s": self.state.now_s, "action": type(action).__name__.lower(), "status": "invalid", "reason": reason})
        return self.public_snapshot()

    def _distance(self, target_zone):
        a, b = self.config.zones[self.state.zone_id], self.config.zones[target_zone]
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def step(self, action):
        s, c = self.state, self.config
        if s.stopped: raise RuntimeError("Trial has ended")
        s.action_count += 1
        if s.action_count > c.max_actions: return self._invalid(action, "max_actions_exceeded")
        if isinstance(action, Stop):
            s.stopped = True; s.log.append({"time_s": s.now_s, "action": "stop", "status": "accepted"}); return self.public_snapshot()
        if isinstance(action, Wait):
            if not math.isfinite(action.duration_s) or action.duration_s <= 0 or s.now_s + action.duration_s > c.horizon_s:
                return self._invalid(action, "invalid_wait")
            before = s.now_s; s.now_s += action.duration_s
            s.log.append({"time_s": before, "action": "wait", "duration_s": action.duration_s, "return_time_s": s.now_s, "status": "accepted"})
            return self.public_snapshot()
        if isinstance(action, Review):
            if action.job_id not in self.jobs: return self._invalid(action, "unknown_job")
            job, runtime = self.jobs[action.job_id], s.jobs[action.job_id]
            if job.release_s > s.now_s or runtime.attempted: return self._invalid(action, "job_not_reviewable")
            if runtime.reviews >= c.max_reviews_per_job or s.review_count >= c.review_limit: return self._invalid(action, "review_limit")
            if s.cost + c.review_cost > c.cost_budget or s.now_s + c.review_delay_s > c.horizon_s: return self._invalid(action, "review_resource_limit")
            before = s.now_s; s.now_s += c.review_delay_s; s.cost += c.review_cost; s.review_count += 1; runtime.reviews += 1
            prefix = (0.2, 0.6, 1.0)[runtime.reviews]
            s.log.append({"time_s": before, "action": "review", "job_id": action.job_id, "return_time_s": s.now_s, "visible_prefix_s": prefix, "status": "accepted"})
            return self.public_snapshot()
        if isinstance(action, Execute):
            if action.job_id not in self.jobs: return self._invalid(action, "unknown_job")
            job, runtime = self.jobs[action.job_id], s.jobs[action.job_id]
            if runtime.attempted or job.release_s > s.now_s: return self._invalid(action, "job_not_executable")
            if any(not s.jobs[p].completed for p in job.predecessors): return self._invalid(action, "predecessor_incomplete")
            if not math.isfinite(action.reserve_s) or action.reserve_s < c.reserve_min_s: return self._invalid(action, "invalid_reservation")
            distance = self._distance(job.zone_id); service_start = s.now_s + distance + c.changeover_s
            action_cost = c.service_fixed_cost + c.reserve_cost_rate * action.reserve_s + c.distance_cost_rate * distance
            if s.cost + action_cost > c.cost_budget or service_start + action.reserve_s > c.horizon_s: return self._invalid(action, "execution_resource_limit")
            before = s.now_s; duration = self._durations[action.job_id]; completion = service_start + duration
            runtime.attempted = True; runtime.service_start_s = service_start; s.cost += action_cost; s.zone_id = job.zone_id
            observed = min(duration, c.horizon_s - service_start)
            overrun = max(observed - action.reserve_s, 0.0); s.observed_overrun_s += overrun
            if completion <= c.horizon_s:
                runtime.completed = True; runtime.completion_s = completion
                idle = max(action.reserve_s - duration, 0.0); s.reserved_idle_s += idle
                s.now_s = min(service_start + max(duration, action.reserve_s), c.horizon_s)
            else:
                runtime.completion_s = None; s.censored_units += 1; s.now_s = c.horizon_s; s.stopped = True; idle = 0.0
            s.log.append({"time_s": before, "action": "execute", "job_id": action.job_id, "reserve_s": action.reserve_s, "distance": distance, "service_start_s": service_start, "completion_s": completion if completion <= c.horizon_s else None, "overrun_s": overrun, "idle_s": idle, "cost": action_cost, "status": "accepted"})
            if s.now_s >= c.horizon_s: s.stopped = True
            return self.public_snapshot()
        return self._invalid(action, "unknown_action")

    def metrics(self) -> dict[str, float | int | bool]:
        c, s = self.config, self.state; total_weight = sum(j.weight for j in self.jobs.values())
        completed_weight = on_time_weight = tardiness = 0.0
        for job_id, job in self.jobs.items():
            runtime = s.jobs[job_id]
            completion = runtime.completion_s if runtime.completed else c.horizon_s
            if runtime.completed:
                completed_weight += job.weight
                if completion <= job.due_s: on_time_weight += job.weight
            tardiness += job.weight * max(completion - job.due_s, 0.0)
        return {
            "weighted_completion_rate": completed_weight / total_weight,
            "weighted_on_time_rate": on_time_weight / total_weight,
            "weighted_tardiness_s": tardiness,
            "cost": s.cost, "observations": s.review_count, "elapsed_s": s.now_s,
            "observed_overrun_s": s.observed_overrun_s, "reserved_idle_s": s.reserved_idle_s,
            "censored_units": s.censored_units, "invalid_actions": s.invalid_actions, "valid_protocol": s.valid_protocol,
        }

