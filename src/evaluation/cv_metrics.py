"""Extended Phase 5 evaluation metrics for 5-fold CV (calibration, ordinal, sequence-level)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from src.cross_validation.runner import CV_TRACK_SPECS, collect_cv_runs_summary
from src.evaluation.metrics import compute_classification_metrics
from src.utils.config import load_config, resolve_path


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over maximum predicted probabilities."""
    if len(y_true) == 0 or y_prob is None or len(y_prob) == 0:
        return 0.0
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y_true))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        cnt = int(np.sum(in_bin))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(acc[in_bin])) - float(np.mean(conf[in_bin])))
    return float(ece)


def compute_multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """Compute multiclass Brier score: mean sum_c (p_ic - y_ic)^2."""
    if len(y_true) == 0 or y_prob is None or len(y_prob) == 0:
        return 0.0
    onehot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def compute_extended_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: np.ndarray | None,
    class_names: Sequence[str],
    is_severity: bool = False,
) -> dict[str, Any]:
    """Compute full classification, calibration, agreement, and exploratory ordinal metrics."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    base = compute_classification_metrics(yt, yp, y_prob, class_names)
    acc = float(base["accuracy"])
    micro_p = float(precision_score(yt, yp, labels=labels, average="micro", zero_division=0))
    micro_r = float(recall_score(yt, yp, labels=labels, average="micro", zero_division=0))
    micro_f1 = float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0))
    mcc = float(matthews_corrcoef(yt, yp)) if len(np.unique(yt)) > 1 or len(np.unique(yp)) > 1 else 0.0
    kappa = float(cohen_kappa_score(yt, yp, labels=labels)) if len(yt) > 1 else 0.0
    if np.isnan(kappa):
        kappa = 0.0

    ll = None
    brier = None
    ece = None
    if y_prob is not None and len(yt) > 0:
        prob = np.asarray(y_prob, dtype=np.float64)
        rs = prob.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        prob = np.clip(prob / rs, 1e-7, 1.0 - 1e-7)
        prob = prob / prob.sum(axis=1, keepdims=True)
        ll = round(float(log_loss(yt, prob, labels=labels)), 6)
        brier = round(compute_multiclass_brier(yt, prob, n_classes), 6)
        ece = round(compute_ece(yt, prob, n_bins=10), 6)

    out = dict(base)
    out.update({
        "error_rate": round(1.0 - acc, 6),
        "micro_precision": round(micro_p, 6),
        "micro_recall": round(micro_r, 6),
        "micro_f1": round(micro_f1, 6),
        "mcc": round(mcc, 6),
        "cohen_kappa": round(kappa, 6),
        "log_loss": ll,
        "brier_score": brier,
        "ece": ece,
    })

    if is_severity:
        qwk = float(cohen_kappa_score(yt, yp, labels=labels, weights="quadratic")) if len(yt) > 1 else 0.0
        if np.isnan(qwk):
            qwk = 0.0
        out["qwk"] = round(qwk, 6)
        out["mae"] = round(float(mean_absolute_error(yt, yp)), 6)
        out["mse"] = round(float(mean_squared_error(yt, yp)), 6)
    else:
        out["qwk"] = None
        out["mae"] = None
        out["mse"] = None
    return out


def aggregate_sequence_level_predictions(pred_df: pd.DataFrame, class_names: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate window-level predictions to unique sequence_id level via mean predicted probabilities."""
    prob_cols = [f"prob_{c}" for c in class_names]
    seq_rows: list[dict[str, Any]] = []
    for seq_id, sdf in pred_df.groupby("group_id", sort=True):
        mean_probs = sdf[prob_cols].to_numpy(dtype=np.float64).mean(axis=0)
        mean_probs = mean_probs / max(1e-12, mean_probs.sum())
        true_idx = int(sdf["true_idx"].iloc[0])
        pred_idx = int(np.argmax(mean_probs))
        row = {
            "sequence_id": str(seq_id),
            "n_windows": int(len(sdf)),
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "true_label": str(class_names[true_idx]),
            "pred_label": str(class_names[pred_idx]),
            "correct": bool(true_idx == pred_idx),
        }
        for c, p in zip(class_names, mean_probs):
            row[f"prob_{c}"] = float(p)
        seq_rows.append(row)
    seq_df = pd.DataFrame(seq_rows)
    probs_mat = seq_df[prob_cols].to_numpy(dtype=np.float64)
    metrics = compute_extended_metrics(seq_df["true_idx"].to_numpy(), seq_df["pred_idx"].to_numpy(), probs_mat, class_names, is_severity=False)
    return seq_df, metrics


def run_cv_evaluation(
    cfg: dict[str, Any] | None = None,
    cv_cfg: dict[str, Any] | None = None,
    output_dir: str | Path = "results/evaluation",
    pub_fig_dir: str | Path = "results/publication_figures",
) -> dict[str, Any]:
    """Compute fold-wise, model-wise (mean+/-std+95%CI), OOF, per-class, and sequence metrics across CV runs."""
    from src.evaluation.cv_plots import generate_all_cv_evaluation_figures

    cv_raw = cv_cfg or load_config("configs/cross_validation.yaml")
    cv_p = cv_raw.get("cross_validation", cv_raw)
    pred_dir = resolve_path(cv_p["paths"]["predictions_dir"])
    runs_df = collect_cv_runs_summary(cv_raw)
    if runs_df.empty:
        raise RuntimeError("No completed CV runs found in results/cross_validation/runs/.")

    out_root = resolve_path(output_dir)
    fig_dir = out_root / "figures"
    pub_dir = resolve_path(pub_fig_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    fold_metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    oof_metric_rows: list[dict[str, Any]] = []
    seq_level_rows: list[dict[str, Any]] = []
    oof_preds_store: dict[tuple[str, str], pd.DataFrame] = {}

    scalar_keys = [
        "accuracy", "error_rate", "balanced_accuracy",
        "macro_precision", "micro_precision", "weighted_precision",
        "macro_recall", "micro_recall", "weighted_recall",
        "macro_f1", "micro_f1", "weighted_f1",
        "macro_specificity", "mcc", "cohen_kappa",
        "roc_auc_ovr", "pr_auc_ovr", "log_loss", "brier_score", "ece",
        "qwk", "mae", "mse",
    ]

    for (track, model_name), grp in runs_df.groupby(["track", "model"], sort=True):
        _, _, class_names, _ = CV_TRACK_SPECS[track]
        prob_cols = [f"prob_{c}" for c in class_names]
        is_sev = track == "oct_severity"
        fold_dfs: list[pd.DataFrame] = []

        for r in grp.sort_values("fold").itertuples(index=False):
            p_path = pred_dir / f"{r.cv_exp_id}_predictions.csv"
            pdf = pd.read_csv(p_path)
            pdf["fold"] = int(r.fold)
            fold_dfs.append(pdf)

            m = compute_extended_metrics(
                pdf["true_idx"].to_numpy(dtype=np.int64),
                pdf["pred_idx"].to_numpy(dtype=np.int64),
                pdf[prob_cols].to_numpy(dtype=np.float64),
                class_names,
                is_severity=is_sev,
            )
            f_row: dict[str, Any] = {
                "cv_exp_id": r.cv_exp_id,
                "track": track,
                "model": model_name,
                "category": r.category,
                "fold": int(r.fold),
                "n_samples": int(m["num_samples"]),
                "best_epoch": int(r.best_epoch),
                "best_val_macro_f1": float(r.best_val_macro_f1),
                "train_duration_sec": float(r.train_duration_sec),
                "inference_latency_ms": float(r.inference_latency_ms),
                "total_params": int(r.total_params),
                "peak_gpu_mem_mb": float(r.peak_gpu_mem_mb),
                "ckpt_size_mb": float(r.ckpt_size_mb),
            }
            for k in scalar_keys:
                f_row[k] = m[k]
            fold_metric_rows.append(f_row)

            for cname, cm_d in m["per_class"].items():
                per_class_rows.append({
                    "track": track,
                    "model": model_name,
                    "fold": int(r.fold),
                    "scope": "fold",
                    "class": cname,
                    **cm_d,
                })

        # Pooled Out-of-Fold (OOF) aggregation across all 5 outer test folds
        oof_df = pd.concat(fold_dfs, ignore_index=True)
        oof_preds_store[(track, model_name)] = oof_df
        oof_m = compute_extended_metrics(
            oof_df["true_idx"].to_numpy(dtype=np.int64),
            oof_df["pred_idx"].to_numpy(dtype=np.int64),
            oof_df[prob_cols].to_numpy(dtype=np.float64),
            class_names,
            is_severity=is_sev,
        )
        oof_row: dict[str, Any] = {
            "track": track,
            "model": model_name,
            "category": grp["category"].iloc[0],
            "n_folds_completed": int(len(grp)),
            "oof_n_samples": int(oof_m["num_samples"]),
            "oof_confusion_matrix_json": json.dumps(oof_m["confusion_matrix"]),
        }
        for k in scalar_keys:
            oof_row[f"oof_{k}"] = oof_m[k]
        oof_metric_rows.append(oof_row)

        for cname, cm_d in oof_m["per_class"].items():
            per_class_rows.append({
                "track": track,
                "model": model_name,
                "fold": 0,
                "scope": "oof",
                "class": cname,
                **cm_d,
            })

        if track == "oct_sequence":
            seq_df, seq_m = aggregate_sequence_level_predictions(oof_df, class_names)
            seq_row: dict[str, Any] = {
                "track": track,
                "model": model_name,
                "category": grp["category"].iloc[0],
                "n_sequences": int(len(seq_df)),
                "confusion_matrix_json": json.dumps(seq_m["confusion_matrix"]),
            }
            for k in scalar_keys:
                if seq_m[k] is not None:
                    seq_row[f"seq_{k}"] = seq_m[k]
            seq_level_rows.append(seq_row)

    fold_df = pd.DataFrame(fold_metric_rows)
    per_cls_df = pd.DataFrame(per_class_rows)
    oof_df_all = pd.DataFrame(oof_metric_rows)
    seq_lvl_df = pd.DataFrame(seq_level_rows)

    # Compute mean, std, 95% t-CI across 5 folds + merge with OOF metrics & compute stats
    summary_rows: list[dict[str, Any]] = []
    agg_metrics = ["accuracy", "error_rate", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_specificity", "mcc", "cohen_kappa", "roc_auc_ovr", "log_loss", "brier_score", "ece"]
    if (fold_df["track"] == "oct_severity").any():
        agg_metrics += ["qwk", "mae", "mse"]

    for (track, model_name), sub in fold_df.groupby(["track", "model"], sort=True):
        row: dict[str, Any] = {
            "track": track,
            "model": model_name,
            "category": sub["category"].iloc[0],
            "folds": int(len(sub)),
            "total_params": int(sub["total_params"].iloc[0]),
            "ckpt_size_mb": round(float(sub["ckpt_size_mb"].mean()), 3),
            "peak_gpu_mem_mb_max": round(float(sub["peak_gpu_mem_mb"].max()), 2),
            "train_time_mean_sec": round(float(sub["train_duration_sec"].mean()), 2),
            "train_time_total_sec": round(float(sub["train_duration_sec"].sum()), 2),
            "inference_latency_ms": round(float(sub["inference_latency_ms"].mean()), 4),
            "best_epoch_mean": round(float(sub["best_epoch"].mean()), 2),
        }
        for mk in agg_metrics:
            vals = sub[mk].dropna().to_numpy(dtype=np.float64)
            if len(vals):
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                se = sd / np.sqrt(len(vals))
                # Exact 95% t-halfwidth for df=n-1 (2.7764 for n=5)
                t_crit = 2.776445105 if len(vals) == 5 else 1.96
                row[f"{mk}_mean"] = round(mu, 6)
                row[f"{mk}_std"] = round(sd, 6)
                row[f"{mk}_ci95_low"] = round(mu - t_crit * se, 6)
                row[f"{mk}_ci95_high"] = round(mu + t_crit * se, 6)
        oof_match = oof_df_all[(oof_df_all["track"] == track) & (oof_df_all["model"] == model_name)].iloc[0]
        row["oof_macro_f1"] = oof_match["oof_macro_f1"]
        row["oof_accuracy"] = oof_match["oof_accuracy"]
        row["oof_mcc"] = oof_match["oof_mcc"]
        row["oof_ece"] = oof_match["oof_ece"]
        summary_rows.append(row)

    model_summary_df = pd.DataFrame(summary_rows).sort_values(["track", "macro_f1_mean"], ascending=[True, False]).reset_index(drop=True)

    fold_df.to_csv(out_root / "cv_fold_metrics.csv", index=False)
    model_summary_df.to_csv(out_root / "cv_model_summary.csv", index=False)
    oof_df_all.to_csv(out_root / "cv_oof_summary.csv", index=False)
    per_cls_df.to_csv(out_root / "cv_per_class_metrics.csv", index=False)
    if not seq_lvl_df.empty:
        seq_lvl_df.to_csv(out_root / "oct_sequence_level_summary.csv", index=False)

    # Track-separated model summaries for clean paper/notebook inspection
    for trk, tdf in model_summary_df.groupby("track"):
        tdf.to_csv(out_root / f"cv_summary_{trk}.csv", index=False)

    figs = generate_all_cv_evaluation_figures(
        runs_df=runs_df,
        fold_df=fold_df,
        model_summary_df=model_summary_df,
        per_cls_df=per_cls_df,
        oof_df_all=oof_df_all,
        oof_preds_store=oof_preds_store,
        seq_lvl_df=seq_lvl_df,
        cv_raw=cv_raw,
        fig_dir=fig_dir,
        pub_dir=pub_dir,
    )

    manifest = {
        "completed_cv_runs": int(len(runs_df)),
        "tracks": sorted(runs_df["track"].unique().tolist()),
        "models_per_track": runs_df.groupby("track")["model"].nunique().to_dict(),
        "figures_generated_count": int(len(figs)),
        "output_dir": str(out_root),
        "publication_figures_dir": str(pub_dir),
    }
    (out_root / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

