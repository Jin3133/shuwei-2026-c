from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForecastOutput:
    mean: torch.Tensor
    log_variance: torch.Tensor | None = None
    reliability_logits: torch.Tensor | None = None

    @property
    def variance(self) -> torch.Tensor | None:
        return None if self.log_variance is None else self.log_variance.exp()

