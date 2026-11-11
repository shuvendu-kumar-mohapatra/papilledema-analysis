"""Reusable PyTorch training engine with AMP, early stopping, resume, and OOM handling."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import build_loss_fn
from src.utils.device import clear_gpu_memory, get_gpu_memory_mb, move_inputs_to_device


def _extract_batch_labels(loader: DataLoader) -> list[int]:
    ds = loader.dataset
    if hasattr(ds, "df") and "label_idx" in ds.df.columns:
        return [int(v) for v in ds.df["label_idx"].tolist()]
    if hasattr(ds, "sequences"):
        return [int(s["label_idx"]) for s in ds.sequences]
    labels = []
    for batch in loader:
        labels.extend(batch[1].tolist())
    return labels


def _pretrain_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    logger: Any | None = None,
) -> list[float]:
    """Train autoencoder reconstruction exclusively on training split (never val/test)."""
    if epochs <= 0 or not hasattr(model, "reconstruction_loss"):
        return []
    opt = torch.optim.AdamW(model.parameters(), lr=lr * 5.0, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=use_amp)
    losses = []
    model.train()
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for batch in train_loader:
            x = move_inputs_to_device(batch[0], device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                rloss = model.reconstruction_loss(x)
            scaler.scale(rloss).backward()
            scaler.step(opt)
            scaler.update()
            bs = x.size(0)
            total += float(rloss.item()) * bs
            n += bs
        avg = total / max(1, n)
        losses.append(avg)
        if logger:
            logger.info("AE pretrain epoch %d/%d | recon_mse=%.5f", ep, epochs, avg)
    return losses


def _forward_and_loss(
    model: nn.Module,
    inputs: Any,
    targets: torch.Tensor,
    criterion: nn.Module,
    recon_weight: float,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast("cuda", enabled=use_amp):
        if hasattr(model, "forward_with_recon") and recon_weight > 0:
            logits, recon_loss = model.forward_with_recon(inputs)
            loss = criterion(logits, targets) + recon_weight * recon_loss
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
    return logits, loss


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float = 1.0,
    recon_weight: float = 0.0,
    logger: Any | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_n = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in loader:
        raw_x, raw_y = batch[0], batch[1]
        targets = raw_y.to(device, non_blocking=(device.type == "cuda"))
        inputs = move_inputs_to_device(raw_x, device)
        bs = int(targets.size(0))

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            try:
                logits, loss = _forward_and_loss(model, inputs, targets, criterion, recon_weight, use_amp)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or bs <= 1:
                    raise
                # Graceful OOM fallback: split current batch into micro-batches of 1 or half
                clear_gpu_memory()
                if logger:
                    logger.warning("CUDA OOM on batch size %d; retrying with micro-batch accumulation.", bs)
                optimizer.zero_grad(set_to_none=True)
                half = max(1, bs // 2)
                logits_parts = []
                loss_accum = 0.0
                for s in range(0, bs, half):
                    e = min(bs, s + half)
                    sub_x = tuple(t[s:e] for t in inputs) if isinstance(inputs, tuple) else inputs[s:e]
                    sub_y = targets[s:e]
                    sub_logits, sub_loss = _forward_and_loss(model, sub_x, sub_y, criterion, recon_weight, use_amp)
                    scaled_loss = sub_loss * ((e - s) / bs)
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    logits_parts.append(sub_logits.detach())
                    loss_accum += float(sub_loss.item()) * (e - s)
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                logits = torch.cat(logits_parts, dim=0)
                loss = torch.tensor(loss_accum / bs, device=device)
        else:
            with torch.no_grad():
                logits, loss = _forward_and_loss(model, inputs, targets, criterion, recon_weight, use_amp)

        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        total_loss += float(loss.item()) * bs
        total_n += bs

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    acc = float(np.mean(y_true == y_pred)) if total_n > 0 else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total_n > 0 else 0.0
    return {
        "loss": total_loss / max(1, total_n),
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    num_classes: int,
    checkpoint_dir: str | Path,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    early_stopping_patience: int = 7,
    use_mixed_precision: bool = True,
    use_class_weights: bool = True,
    scheduler_type: str = "cosine",
    autoencoder_pretrain_epochs: int = 5,
    recon_loss_weight: float = 0.1,
    resume: bool = True,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Train a model sequentially, selecting best checkpoint by validation macro F1."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    use_amp = bool(use_mixed_precision and device.type == "cuda")
    model = model.to(device)

    train_labels = _extract_batch_labels(loaders["train"])
    criterion = build_loss_fn(train_labels, num_classes, use_class_weights=use_class_weights, device=device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    else:
        scheduler = None
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=use_amp)

    start_epoch = 1
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, Any]] = []
    ae_losses: list[float] = []

    if resume and last_path.exists():
        ckpt = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if use_amp and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        patience_counter = int(ckpt.get("patience_counter", 0))
        history = list(ckpt.get("history", []))
        if logger:
            logger.info("Resuming from %s at epoch %d (best val macro F1=%.4f)", last_path, start_epoch, best_val_f1)
    elif hasattr(model, "reconstruction_loss"):
        ae_losses = _pretrain_autoencoder(
            model, loaders["train"], device, epochs=autoencoder_pretrain_epochs,
            lr=lr, weight_decay=weight_decay, use_amp=use_amp, logger=logger,
        )

    recon_w = recon_loss_weight if hasattr(model, "forward_with_recon") else 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    t_start = time.perf_counter()
    already_stopped = early_stopping_patience > 0 and patience_counter >= early_stopping_patience
    for epoch in ([] if already_stopped else range(start_epoch, epochs + 1)):
        ep_start = time.perf_counter()
        tr = _run_epoch(
            model, loaders["train"], criterion, device, use_amp,
            optimizer=optimizer, scaler=scaler if use_amp else None,
            grad_clip_norm=grad_clip_norm, recon_weight=recon_w, logger=logger,
        )
        va = _run_epoch(
            model, loaders["val"], criterion, device, use_amp,
            optimizer=None, recon_weight=0.0, logger=logger,
        )

        cur_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(va["macro_f1"])
            else:
                scheduler.step()

        mem = get_gpu_memory_mb(device)
        ep_dur = time.perf_counter() - ep_start
        row = {
            "epoch": epoch,
            "train_loss": round(tr["loss"], 6),
            "val_loss": round(va["loss"], 6),
            "train_accuracy": round(tr["accuracy"], 6),
            "val_accuracy": round(va["accuracy"], 6),
            "train_macro_f1": round(tr["macro_f1"], 6),
            "val_macro_f1": round(va["macro_f1"], 6),
            "lr": cur_lr,
            "epoch_duration_sec": round(ep_dur, 3),
            "gpu_peak_memory_mb": mem["peak_allocated_mb"],
        }
        history.append(row)

        improved = va["macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = va["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(best_path, {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_f1": best_val_f1,
            })
        else:
            patience_counter += 1

        save_checkpoint(last_path, {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if use_amp else None,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history": history,
        })

        if logger:
            logger.info(
                "Epoch %d/%d | tr_loss=%.4f va_loss=%.4f | tr_f1=%.4f va_f1=%.4f | lr=%.2e | %.2fs",
                epoch, epochs, tr["loss"], va["loss"], tr["macro_f1"], va["macro_f1"], cur_lr, ep_dur,
            )

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            if logger:
                logger.info("Early stopping triggered at epoch %d (best epoch=%d).", epoch, best_epoch)
            break

    # Ensure best.pt exists even if epochs == 0 or resume skipped loop
    if not best_path.exists():
        save_checkpoint(best_path, {"epoch": 0, "model_state": model.state_dict(), "best_val_f1": best_val_f1})

    total_dur = (
        round(sum(float(r.get("epoch_duration_sec", 0.0)) for r in history), 3)
        if history
        else round(time.perf_counter() - t_start, 3)
    )
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(ckpt_dir / "history.csv", index=False)

    return {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "history": history,
        "train_duration_sec": total_dur,
        "ae_pretrain_losses": ae_losses,
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
    }
