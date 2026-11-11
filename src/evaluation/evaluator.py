"""Test-set model evaluation, latency benchmarking, and per-sample error analysis."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics
from src.models.model_factory import count_parameters
from src.utils.device import move_inputs_to_device


def evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: str | Path,
    class_names: list[str],
    device: torch.device,
    predictions_dir: str | Path,
    exp_id: str,
    use_amp: bool = True,
    low_conf_threshold: float = 0.6,
    high_conf_error_threshold: float = 0.8,
) -> dict[str, Any]:
    """Load best validation-selected checkpoint and evaluate on untouched test set."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    amp_enabled = bool(use_amp and device.type == "cuda")
    all_probs: list[np.ndarray] = []
    all_targets: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    # Warm-up 1 batch if available for accurate timing
    first_batch = next(iter(test_loader), None)
    if first_batch is not None:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
            _ = model(move_inputs_to_device(first_batch[0], device))
        if device.type == "cuda":
            torch.cuda.synchronize()

    infer_start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            raw_x, raw_y = batch[0], batch[1]
            meta = batch[2] if len(batch) > 2 else None
            inputs = move_inputs_to_device(raw_x, device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            targets_list = raw_y.tolist()
            all_targets.extend(targets_list)

            bs = len(targets_list)
            for i in range(bs):
                if isinstance(meta, dict):
                    meta_rows.append({
                        "sample_id": str(meta["sample_id"][i]),
                        "image_path": str(meta["image_path"][i]),
                        "group_id": str(meta["group_id"][i]),
                    })
                else:
                    idx_global = len(meta_rows)
                    meta_rows.append({
                        "sample_id": f"sample_{idx_global}",
                        "image_path": "",
                        "group_id": f"group_{idx_global}",
                    })

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_elapsed = max(1e-6, time.perf_counter() - infer_start)

    prob_mat = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(class_names)))
    y_true = np.asarray(all_targets, dtype=np.int64)
    y_pred = np.argmax(prob_mat, axis=1) if len(prob_mat) else np.zeros((0,), dtype=np.int64)
    confidences = np.max(prob_mat, axis=1) if len(prob_mat) else np.zeros((0,))

    metrics = compute_classification_metrics(y_true, y_pred, prob_mat, class_names)
    params = count_parameters(model)
    n_samples = len(y_true)

    # Save predictions and error analysis CSVs
    pred_dir = Path(predictions_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_records = []
    for i in range(n_samples):
        t_idx, p_idx = int(y_true[i]), int(y_pred[i])
        conf = float(confidences[i])
        rec = {
            **meta_rows[i],
            "true_idx": t_idx,
            "pred_idx": p_idx,
            "true_label": class_names[t_idx],
            "pred_label": class_names[p_idx],
            "correct": bool(t_idx == p_idx),
            "confidence": round(conf, 6),
            "low_confidence": bool(conf < low_conf_threshold),
            "high_confidence_error": bool((t_idx != p_idx) and (conf >= high_conf_error_threshold)),
        }
        for c_i, cname in enumerate(class_names):
            rec[f"prob_{cname}"] = round(float(prob_mat[i, c_i]), 6)
        pred_records.append(rec)

    pred_df = pd.DataFrame(pred_records)
    pred_csv = pred_dir / f"{exp_id}_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    errors_df = pred_df[~pred_df["correct"]].copy()
    errors_csv = pred_dir / f"{exp_id}_errors.csv"
    errors_df.to_csv(errors_csv, index=False)

    confused_pairs = Counter(
        f"{r['true_label']} -> {r['pred_label']}" for _, r in errors_df.iterrows()
    )
    error_summary = {
        "total_errors": int(len(errors_df)),
        "error_rate": round(float(len(errors_df) / max(1, n_samples)), 6),
        "low_confidence_count": int(pred_df["low_confidence"].sum()) if n_samples else 0,
        "high_confidence_error_count": int(errors_df["high_confidence_error"].sum()) if not errors_df.empty else 0,
        "confused_pairs": dict(confused_pairs.most_common()),
        "predictions_csv": str(pred_csv),
        "errors_csv": str(errors_csv),
    }

    return {
        "exp_id": exp_id,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "best_val_macro_f1": float(ckpt.get("best_val_f1", float("nan"))),
        "metrics": metrics,
        "parameters": params,
        "inference_total_sec": round(infer_elapsed, 4),
        "inference_latency_ms": round((infer_elapsed / max(1, n_samples)) * 1000.0, 4),
        "throughput_samples_per_sec": round(n_samples / infer_elapsed, 2),
        "error_analysis": error_summary,
    }
