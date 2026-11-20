from __future__ import annotations

import torch

from .models.outputs import ForecastOutput


@torch.no_grad()
def trajectory_metrics(output: ForecastOutput, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    distance = torch.linalg.vector_norm(output.mean - target, dim=-1)
    valid = distance[mask]
    result = {"ade_m": valid.mean().item(), "rmse_m": torch.sqrt((valid**2).mean()).item()}
    final_valid = mask[:, -1]
    result["fde_m"] = distance[:, -1][final_valid].mean().item() if final_valid.any() else float("nan")
    if output.variance is not None:
        sigma = output.variance.sqrt()
        covered = ((target >= output.mean - 1.96 * sigma) & (target <= output.mean + 1.96 * sigma)).all(dim=-1)
        result["coverage_95"] = covered[mask].float().mean().item()
        result["mean_std_m"] = sigma.mean(dim=-1)[mask].mean().item()
    return result

