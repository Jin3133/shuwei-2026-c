from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservationContext:
    missing_elapsed_s: float
    request_pending: bool
    requests_used: int
    request_budget: int
    return_delay_s: float


@dataclass(frozen=True)
class ObservationDecision:
    request: bool
    uncertainty_radius_m: float
    reason: str


class FixedIntervalPolicy:
    def __init__(self, interval_s: float): self.interval_s = interval_s

    def decide(self, variance: torch.Tensor | None, context: ObservationContext) -> ObservationDecision:
        allowed = not context.request_pending and context.requests_used < context.request_budget
        request = allowed and context.missing_elapsed_s >= self.interval_s
        return ObservationDecision(request, float("nan"), "fixed_interval" if request else "wait")


class UncertaintyTriggerPolicy:
    """Request when calibrated 3D uncertainty radius exceeds a frozen threshold."""

    def __init__(self, radius_threshold_m: float, z_value: float = 1.96):
        self.radius_threshold_m, self.z_value = radius_threshold_m, z_value

    def decide(self, variance: torch.Tensor | None, context: ObservationContext) -> ObservationDecision:
        if variance is None: raise ValueError("UncertaintyTriggerPolicy requires predictive variance")
        radius = self.z_value * torch.sqrt(torch.clamp(variance, min=0)).norm(dim=-1).max().item()
        allowed = not context.request_pending and context.requests_used < context.request_budget
        request = allowed and radius > self.radius_threshold_m
        return ObservationDecision(request, radius, "uncertainty_threshold" if request else "wait")

