"""Causal artificial-masking replay for Problem 2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReturnedObservation:
    source_index: int
    timestamp_s: float
    state: np.ndarray


@dataclass
class PendingRequest:
    source_index: int
    return_at_s: float


class MissingObservationReplay:
    """Expose a trajectory sequentially without leaking masked references.

    The private reference array is never returned inside the mask except after a
    requested, existing observation has completed its fixed delay.
    """

    def __init__(self, timestamps_s, reference_states, reference_ok, mask_start_s: float, mask_end_s: float, return_delay_s: float, request_budget: int):
        self._time = np.asarray(timestamps_s, dtype=float)
        self._reference = np.asarray(reference_states, dtype=float)
        self._ok = np.asarray(reference_ok, dtype=bool)
        if self._time.ndim != 1 or self._reference.shape[0] != len(self._time) or self._ok.shape != self._time.shape:
            raise ValueError("Replay arrays have inconsistent shapes")
        if np.any(np.diff(self._time) < 0) or not mask_start_s < mask_end_s:
            raise ValueError("Timestamps must be monotonic and mask interval non-empty")
        self.mask_start_s, self.mask_end_s = mask_start_s, mask_end_s
        self.return_delay_s, self.request_budget = return_delay_s, request_budget
        self.index = 0; self.requests_used = 0; self.pending: PendingRequest | None = None

    @property
    def now_s(self) -> float: return float(self._time[self.index])

    @property
    def done(self) -> bool: return self.index >= len(self._time) - 1

    def visible_observation(self) -> ReturnedObservation | None:
        hidden = self.mask_start_s <= self.now_s < self.mask_end_s
        if hidden or not self._ok[self.index]: return None
        return ReturnedObservation(self.index, self.now_s, self._reference[self.index].copy())

    def request(self) -> None:
        if self.pending is not None: raise ValueError("A request is already pending")
        if self.requests_used >= self.request_budget: raise ValueError("Request budget exhausted")
        if not self._ok[self.index]: raise ValueError("The source record has no qualified observation at this time")
        self.requests_used += 1
        self.pending = PendingRequest(self.index, self.now_s + self.return_delay_s)

    def advance(self) -> ReturnedObservation | None:
        if self.done: return None
        self.index += 1
        if self.pending is not None and self.now_s + 1e-12 >= self.pending.return_at_s:
            idx = self.pending.source_index; self.pending = None
            return ReturnedObservation(idx, float(self._time[idx]), self._reference[idx].copy())
        return None

    def private_reference_for_scoring(self) -> np.ndarray:
        """Evaluation-only accessor; never pass its result to a policy."""
        return self._reference.copy()

