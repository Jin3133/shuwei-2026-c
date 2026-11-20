"""Small, auditable scheduling policies; no hidden-duration access."""
from __future__ import annotations

from dataclasses import dataclass

from .simulator import Execute, Job, Stop, Wait


@dataclass(frozen=True)
class DurationPrediction:
    q50_s: float
    q80_s: float
    q90_s: float


class RiskAwarePriorityPolicy:
    def __init__(self, reservation_quantile: str = "q80_s", deadline_weight: float = 1.0):
        self.reservation_quantile = reservation_quantile; self.deadline_weight = deadline_weight

    def act(self, snapshot: dict, jobs: dict[str, Job], predictions: dict[str, DurationPrediction]):
        now, runtime = snapshot["now_s"], snapshot["jobs"]
        feasible = [j for j in jobs.values() if j.release_s <= now and not runtime[j.job_id]["attempted"] and all(runtime[p]["completed"] for p in j.predecessors)]
        if not feasible:
            future = [j.release_s for j in jobs.values() if not runtime[j.job_id]["attempted"] and j.release_s > now]
            return Wait(min(future) - now) if future else Stop()
        def score(job):
            duration = getattr(predictions[job.job_id], self.reservation_quantile)
            slack = max(job.due_s - now - duration, 1e-3)
            return job.weight / duration * (1 + self.deadline_weight / slack)
        chosen = max(feasible, key=score)
        return Execute(chosen.job_id, max(0.2, getattr(predictions[chosen.job_id], self.reservation_quantile)))

