"""Descriptive statistics across multiple random seeds."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from scipy import stats


def summarize_array(values: Sequence[float], alpha: float = 0.05) -> dict[str, float | int | None]:
    """Compute mean, std, median, min, max, IQR, and t-interval 95% CI."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    n = int(len(arr))
    if n == 0:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "min": None, "max": None, "iqr": None, "ci95_low": None, "ci95_high": None,
        }
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    if n >= 2:
        std = float(np.std(arr, ddof=1))
        se = std / np.sqrt(n)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        std = 0.0
        ci_low = None
        ci_high = None

    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "median": round(median, 6),
        "min": round(vmin, 6),
        "max": round(vmax, 6),
        "iqr": round(iqr, 6),
        "ci95_low": round(ci_low, 6) if ci_low is not None else None,
        "ci95_high": round(ci_high, 6) if ci_high is not None else None,
    }


def compute_descriptive_statistics(
    runs_df: pd.DataFrame,
    group_cols: Sequence[str] = ("track", "model"),
    metric_cols: Sequence[str] = (
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "roc_auc_ovr",
        "pr_auc_ovr",
        "train_duration_sec",
        "inference_latency_ms",
        "throughput_samples_per_sec",
        "total_params",
    ),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize each metric across runs/seeds for each (track, model)."""
    if runs_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, gdf in runs_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        for m in metric_cols:
            if m not in gdf.columns:
                continue
            s = summarize_array(gdf[m].tolist(), alpha=alpha)
            rows.append({
                **key_dict,
                "metric": m,
                **s,
                "note": "Seed variance on fixed split; NOT independent patient cohorts." if s["n"] > 1 else "Single run (n=1); variability unknown.",
            })
    return pd.DataFrame(rows)

