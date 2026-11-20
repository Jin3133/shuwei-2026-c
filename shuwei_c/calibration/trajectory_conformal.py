"""Split-conformal calibration for multi-horizon 3D trajectory intervals."""
from __future__ import annotations

import torch


class HorizonConformalCalibrator:
    """Calibrate one multiplicative radius per forecast horizon on validation data."""

    def __init__(self, coverage: float = 0.9, eps: float = 1e-6):
        if not 0 < coverage < 1: raise ValueError("coverage must lie in (0,1)")
        self.coverage, self.eps, self.scale = coverage, eps, None

    def fit(self, mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mean.shape != target.shape or variance.shape != target.shape or mask.shape != target.shape[:2]:
            raise ValueError("Calibration tensors have incompatible shapes")
        error_radius = torch.linalg.vector_norm(target - mean, dim=-1)
        predicted_radius = torch.linalg.vector_norm(torch.sqrt(torch.clamp(variance, min=self.eps)), dim=-1)
        scores = error_radius / predicted_radius.clamp_min(self.eps)
        scales = []
        for horizon in range(scores.shape[1]):
            values = scores[:, horizon][mask[:, horizon]]
            if not len(values): raise ValueError(f"No calibration samples at horizon {horizon}")
            n = len(values); q = min(1.0, (self.coverage * (n + 1)).__ceil__() / n)
            scales.append(torch.quantile(values, q))
        self.scale = torch.stack(scales)
        return self

    def radius(self, variance: torch.Tensor) -> torch.Tensor:
        if self.scale is None: raise RuntimeError("Calibrator has not been fitted")
        base = torch.linalg.vector_norm(torch.sqrt(torch.clamp(variance, min=self.eps)), dim=-1)
        return base * self.scale.to(device=variance.device, dtype=variance.dtype)[None, :]

    def state_dict(self) -> dict:
        if self.scale is None: raise RuntimeError("Calibrator has not been fitted")
        return {"coverage": self.coverage, "eps": self.eps, "scale": self.scale.cpu()}
