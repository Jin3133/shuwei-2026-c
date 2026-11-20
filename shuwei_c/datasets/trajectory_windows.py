from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryWindowDataset(Dataset):
    """Q1/Q2 window dataset.

    NPZ contract:
      x: float32 [N, L, F], canonical features [dx,dy,dz,vx,vy,vz,quality,hand,...]
      y: float32 [N, H, 3], future displacement relative to the final observed position
      target_mask: optional bool [N, H], true where ground truth is valid
      episode_uuid/hand/start_s: optional metadata arrays
    """

    def __init__(self, path: str | Path):
        data = np.load(Path(path), allow_pickle=False)
        self.x = torch.from_numpy(np.asarray(data["x"], dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(data["y"], dtype=np.float32))
        mask = data["target_mask"] if "target_mask" in data else np.ones(self.y.shape[:2], dtype=bool)
        self.target_mask = torch.from_numpy(np.asarray(mask, dtype=bool))
        reliability = data["reliability_target"] if "reliability_target" in data else None
        self.reliability_target = None if reliability is None else torch.from_numpy(np.asarray(reliability, dtype=np.float32))
        self._validate()

    def _validate(self) -> None:
        if self.x.ndim != 3 or self.y.ndim != 3:
            raise ValueError(f"Expected x[N,L,F], y[N,H,D]; got {self.x.shape}, {self.y.shape}")
        if len(self.x) != len(self.y) or self.target_mask.shape != self.y.shape[:2]:
            raise ValueError("x, y and target_mask have inconsistent sample/horizon dimensions")
        if self.y.shape[-1] != 3:
            raise ValueError("The current trajectory target must have three spatial dimensions")
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.y).all():
            raise ValueError("Window tensors must be finite; encode missingness as features/masks")
        if self.reliability_target is not None and self.reliability_target.shape != self.x.shape[:2]:
            raise ValueError("reliability_target must have shape [N,L]")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"x": self.x[index], "y": self.y[index], "target_mask": self.target_mask[index]}
        if self.reliability_target is not None:
            item["reliability_target"] = self.reliability_target[index]
        return item


def make_synthetic_windows(n: int, length: int, horizon: int, input_dim: int, seed: int = 2026):
    """Create kinematically plausible windows for tests only."""
    rng = np.random.default_rng(seed)
    dt = 1 / 30
    velocity = rng.normal(0, 0.25, size=(n, 1, 3)).astype(np.float32)
    acceleration = rng.normal(0, 0.05, size=(n, 1, 3)).astype(np.float32)
    history_t = np.arange(length, dtype=np.float32)[None, :, None] * dt
    future_t = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None] * dt
    position = velocity * history_t + 0.5 * acceleration * history_t**2
    x = np.zeros((n, length, input_dim), dtype=np.float32)
    x[:, :, :3] = position - position[:, :1]
    x[:, :, 3:6] = velocity + acceleration * history_t
    if input_dim > 6:
        x[:, :, 6] = 1.0
    y = (velocity + acceleration * history_t[:, -1:]) * future_t + 0.5 * acceleration * future_t**2
    mask = np.ones((n, horizon), dtype=bool)
    return x, y.astype(np.float32), mask

