from __future__ import annotations

import torch
from torch import nn

from .outputs import ForecastOutput


class LastObservationBaseline(nn.Module):
    """Zero future displacement because targets are relative to the last observation."""

    def __init__(self, horizon: int, target_dim: int = 3):
        super().__init__(); self.horizon = horizon; self.target_dim = target_dim

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        return ForecastOutput(x.new_zeros((len(x), self.horizon, self.target_dim)))


class ConstantVelocityBaseline(nn.Module):
    def __init__(self, horizon: int, sample_period_s: float, velocity_indices=(3, 4, 5)):
        super().__init__(); self.horizon = horizon; self.dt = sample_period_s; self.velocity_indices = velocity_indices

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        velocity = x[:, -1, list(self.velocity_indices)]
        time = torch.arange(1, self.horizon + 1, device=x.device, dtype=x.dtype) * self.dt
        return ForecastOutput(velocity[:, None, :] * time[None, :, None])


class FilterPyKalmanBaseline(nn.Module):
    """Constant-velocity 3D Kalman baseline backed by FilterPy.

    The first three input features are positions relative to the window origin.
    Measurements marked unavailable by quality_index are skipped.
    """

    def __init__(self, horizon: int, sample_period_s: float, quality_index: int = 6, process_variance: float = 1e-3, measurement_variance: float = 1e-3):
        super().__init__(); self.horizon = horizon; self.dt = sample_period_s; self.quality_index = quality_index
        self.process_variance = process_variance; self.measurement_variance = measurement_variance

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        try:
            import numpy as np
            from filterpy.kalman import KalmanFilter
        except ImportError as exc:
            raise RuntimeError("Kalman baseline requires: pip install '.[kalman]'") from exc
        means, log_vars = [], []
        for sample in x.detach().cpu().numpy():
            kf = KalmanFilter(dim_x=6, dim_z=3)
            kf.F = np.block([[np.eye(3), self.dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
            kf.H = np.block([np.eye(3), np.zeros((3, 3))])
            kf.Q = np.eye(6) * self.process_variance
            kf.R = np.eye(3) * self.measurement_variance
            kf.P = np.eye(6)
            kf.x[:3] = sample[0, :3]
            if sample.shape[1] >= 6: kf.x[3:] = sample[0, 3:6]
            for row in sample:
                kf.predict()
                if row[self.quality_index] > 0: kf.update(row[:3])
            origin, sample_means, sample_vars = kf.x[:3].copy(), [], []
            for _ in range(self.horizon):
                kf.predict(); sample_means.append(kf.x[:3] - origin); sample_vars.append(np.diag(kf.P[:3, :3]))
            means.append(np.asarray(sample_means)); log_vars.append(np.log(np.maximum(np.asarray(sample_vars), 1e-8)))
        return ForecastOutput(
            torch.as_tensor(np.asarray(means), device=x.device, dtype=x.dtype),
            torch.as_tensor(np.asarray(log_vars), device=x.device, dtype=x.dtype),
        )

