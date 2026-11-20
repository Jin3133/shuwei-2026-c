from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .models.outputs import ForecastOutput


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand_as(values)
    return values[expanded].mean()


def forecast_loss(output: ForecastOutput, target: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse": return masked_mean((output.mean - target) ** 2, mask)
    if name == "gaussian_nll":
        if output.log_variance is None: raise ValueError("gaussian_nll requires a probabilistic model")
        nll = 0.5 * (math.log(2 * math.pi) + output.log_variance + (target - output.mean) ** 2 / output.log_variance.exp())
        return masked_mean(nll, mask)
    raise ValueError(f"Unknown loss: {name}")


def joint_forecast_loss(output: ForecastOutput, target: torch.Tensor, target_mask: torch.Tensor, reliability_target: torch.Tensor | None, reliability_weight: float = 0.2, horizon_decay: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gaussian trajectory loss plus optional reliability supervision.

    horizon_decay > 0 gives near-term steps larger weight without discarding
    long-horizon supervision.
    """
    if output.log_variance is None:
        trajectory = masked_mean((output.mean - target) ** 2, target_mask)
    else:
        per_element = 0.5 * (math.log(2 * math.pi) + output.log_variance + (target - output.mean) ** 2 / output.log_variance.exp())
        if horizon_decay > 0:
            steps = torch.arange(target.shape[1], device=target.device, dtype=target.dtype)
            per_element = per_element * torch.exp(-horizon_decay * steps)[None, :, None]
        trajectory = masked_mean(per_element, target_mask)
    reliability = target.new_zeros(())
    if reliability_target is not None:
        if output.reliability_logits is None: raise ValueError("Reliability target supplied to a model without reliability head")
        reliability = F.binary_cross_entropy_with_logits(output.reliability_logits, reliability_target)
    total = trajectory + reliability_weight * reliability
    return total, {"trajectory": trajectory.detach(), "reliability": reliability.detach()}

