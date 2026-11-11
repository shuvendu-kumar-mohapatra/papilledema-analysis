"""Grouped & stratified bootstrap confidence intervals respecting cluster/sequence dependence."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


def grouped_bootstrap_ci(
    pred_df: pd.DataFrame,
    class_names: Sequence[str],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    group_col: str = "group_id",
) -> list[dict[str, Any]]:
    """Compute stratified grouped bootstrap 95% CIs on test-set predictions.

    Resamples intact cluster/sequence IDs within each primary class to avoid
    inflating effective sample size across near-duplicate images or adjacent scans.
    """
    if pred_df.empty:
        return []

    df = pred_df.reset_index(drop=True)
    if group_col not in df.columns:
        df[group_col] = [f"sample_{i}" for i in range(len(df))]

    # Map each group to its row indices and primary true label
    groups_by_cls: dict[int, list[np.ndarray]] = {}
    for _, gdf in df.groupby(group_col, sort=False):
        primary_cls = int(gdf["true_idx"].mode().iloc[0])
        groups_by_cls.setdefault(primary_cls, []).append(gdf.index.to_numpy())

    n_groups_total = sum(len(v) for v in groups_by_cls.values())
    min_cls_groups = min(len(v) for v in groups_by_cls.values()) if groups_by_cls else 0
    warning = ""
    if min_cls_groups < 2:
        warning = (
            f"At least one class has <2 independent groups in test (min_groups={min_cls_groups}); "
            "grouped bootstrap reflects limited cluster diversity."
        )
    elif n_groups_total < len(df):
        warning = f"Grouped resampling over {n_groups_total} clusters/sequences ({len(df)} samples)."

    y_true_all = df["true_idx"].to_numpy(dtype=np.int64)
    y_pred_all = df["pred_idx"].to_numpy(dtype=np.int64)
    labels = list(range(len(class_names)))

    def _calc(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
        }

    point = _calc(y_true_all, y_pred_all)
    rng = np.random.default_rng(seed)
    boot_vals: dict[str, list[float]] = {k: [] for k in point}

    for _ in range(n_resamples):
        picked_idx = []
        for cls_idx, grp_arrays in groups_by_cls.items():
            k = len(grp_arrays)
            chosen = rng.integers(0, k, size=k)
            for c_i in chosen:
                picked_idx.append(grp_arrays[c_i])
        idx_arr = np.concatenate(picked_idx)
        m = _calc(y_true_all[idx_arr], y_pred_all[idx_arr])
        for k, v in m.items():
            boot_vals[k].append(v)

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    results = []
    for metric, pt in point.items():
        arr = np.asarray(boot_vals[metric], dtype=np.float64)
        results.append({
            "metric": metric,
            "point_estimate": round(pt, 6),
            "ci95_low": round(float(np.percentile(arr, lo_q)), 6),
            "ci95_high": round(float(np.percentile(arr, hi_q)), 6),
            "n_samples": int(len(df)),
            "n_groups": int(n_groups_total),
            "n_resamples": int(n_resamples),
            "warning": warning,
        })
    return results

