"""Confidence interval forest plots and statistical significance p-value heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.training_plots import _save_fig


def plot_confidence_intervals(
    ci_df: pd.DataFrame,
    figures_dir: str | Path,
    metric: str = "macro_f1",
) -> Path | None:
    """Plot point estimates with grouped bootstrap 95% confidence intervals."""
    if ci_df.empty:
        return None
    sub = ci_df[ci_df["metric"] == metric].copy()
    if sub.empty:
        return None
    sub["label"] = sub["track"].astype(str) + " | " + sub["model"].astype(str)
    sub = sub.sort_values(["track", "point_estimate"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.4 * len(sub) + 1.5)))
    y = np.arange(len(sub))
    pts = sub["point_estimate"].to_numpy(dtype=float)
    lo = np.maximum(0.0, pts - sub["ci95_low"].to_numpy(dtype=float))
    hi = np.maximum(0.0, sub["ci95_high"].to_numpy(dtype=float) - pts)
    ax.errorbar(pts, y, xerr=[lo, hi], fmt="o", color="#2b6cb0", ecolor="#2b6cb0", capsize=4, markersize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (95% Grouped Bootstrap CI)")
    ax.set_title("Model Performance with 95% Confidence Intervals", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    return _save_fig(fig, Path(figures_dir) / "confidence_intervals.png")


def plot_statistical_significance_heatmap(
    pairwise_df: pd.DataFrame,
    figures_dir: str | Path,
    track: str | None = None,
    filename: str = "statistical_significance_heatmap.png",
) -> Path | None:
    """Plot Holm-adjusted pairwise p-value heatmap (preferring McNemar or Paired t-test)."""
    if pairwise_df.empty:
        return None
    # Prefer McNemar test for test-prediction comparison; fallback to Paired t-test
    mcn = pairwise_df[pairwise_df["test_used"].astype(str).str.startswith("McNemar")]
    sub = mcn if not mcn.empty else pairwise_df[pairwise_df["test_used"].astype(str).str.startswith("Paired")]
    if sub.empty:
        return None
    primary_track = track if track is not None else ("fundus_3class" if "fundus_3class" in sub["track"].values else sub["track"].iloc[0])
    sub = sub[sub["track"] == primary_track]
    if sub.empty:
        return None
    models = sorted(set(sub["model_a"]).union(set(sub["model_b"])))
    if len(models) < 2:
        return None

    mat = np.ones((len(models), len(models)), dtype=float)
    idx_map = {m: i for i, m in enumerate(models)}
    for _, r in sub.iterrows():
        p = r["p_adjusted"]
        if pd.isna(p):
            continue
        i, j = idx_map[r["model_a"]], idx_map[r["model_b"]]
        mat[i, j] = float(p)
        mat[j, i] = float(p)

    fig, ax = plt.subplots(figsize=(max(5.5, 0.75 * len(models) + 1.5), max(4.8, 0.7 * len(models) + 1.2)))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=0.20)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(models)):
            txt = "—" if i == j else f"{mat[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="Holm-adjusted p-value")
    ax.set_title(f"Pairwise Significance Heatmap ({primary_track})", fontweight="bold")
    return _save_fig(fig, Path(figures_dir) / filename)

