from __future__ import annotations

import json
from pathlib import Path

import torch

from .losses import forecast_loss, joint_forecast_loss
from .metrics import trajectory_metrics
from .models.outputs import ForecastOutput


def move_batch(batch, device): return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, loss_name="mse"):
    model.eval(); losses, outputs, log_variances, targets, masks = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device); output = model(batch["x"])
        effective_loss = loss_name if output.log_variance is not None else "mse"
        losses.append(forecast_loss(output, batch["y"], batch["target_mask"], effective_loss).item())
        outputs.append(output.mean.cpu())
        if output.log_variance is not None:
            log_variances.append(output.log_variance.cpu())
        targets.append(batch["y"].cpu()); masks.append(batch["target_mask"].cpu())
    merged = ForecastOutput(torch.cat(outputs), torch.cat(log_variances) if log_variances else None)
    result = trajectory_metrics(merged, torch.cat(targets), torch.cat(masks))
    result["loss"] = sum(losses) / len(losses); return result


def fit(model, train_loader, val_loader, config, device):
    train_cfg = config["training"]; checkpoint = Path(train_cfg["checkpoint_dir"]); checkpoint.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler_cfg = train_cfg.get("lr_scheduler")
    scheduler = None if not scheduler_cfg else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 2),
        min_lr=scheduler_cfg.get("min_lr", 1e-5),
    )
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device); optimizer.zero_grad(set_to_none=True)
            output = model(batch["x"])
            if train_cfg["loss"] == "joint_gaussian_nll":
                loss, _ = joint_forecast_loss(output, batch["y"], batch["target_mask"], batch.get("reliability_target"), train_cfg.get("reliability_weight", 0.2), train_cfg.get("horizon_decay", 0.0))
            else:
                loss = forecast_loss(output, batch["y"], batch["target_mask"], train_cfg["loss"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"]); optimizer.step(); running += loss.item()
        eval_loss = "gaussian_nll" if train_cfg["loss"] == "joint_gaussian_nll" else train_cfg["loss"]
        metrics = evaluate(model, val_loader, device, eval_loss)
        if scheduler is not None:
            scheduler.step(metrics["loss"])
        metrics.update(epoch=epoch, train_loss=running / len(train_loader), learning_rate=optimizer.param_groups[0]["lr"]); history.append(metrics)
        if metrics["loss"] < best:
            best, stale = metrics["loss"], 0
            torch.save({"model": model.state_dict(), "config": config, "metrics": metrics}, checkpoint / "best.pt")
        else:
            stale += 1
            if stale >= train_cfg["patience"]: break
    (checkpoint / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return history

