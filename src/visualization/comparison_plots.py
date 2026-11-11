"""Dataset distribution, model family comparison, per-class, and compute cost plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.training_plots import _save_fig


def plot_dataset_class_distribution(
    fundus_df: pd.DataFrame,
    oct_df: pd.DataFrame,
    figures_dir: str | Path,
) -> Path:
    """Plot audited class distributions separately for Fundus, OCT 3-class, and OCT severity."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [
        (axes[0], fundus_df["label"].value_counts() if not fundus_df.empty else pd.Series(dtype=int), "Kim Fundus (Retained)"),
        (axes[1], oct_df["label_3class"].value_counts() if not oct_df.empty else pd.Series(dtype=int), "Original OCT 3-Class (B-scans)"),
        (axes[2], oct_df[oct_df["label_severity"].astype(str) != ""]["label_severity"].value_counts() if not oct_df.empty else pd.Series(dtype=int), "Original OCT Severity (Exploratory)"),
    ]
    colors = ["#2b6cb0", "#c53030", "#2f855a", "#b7791f"]
    for ax, counts, title in panels:
        if counts.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        counts = counts.sort_index()
        bars = ax.bar(counts.index, counts.values, color=colors[: len(counts)], alpha=0.88)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{int(b.get_height())}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Images")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    return _save_fig(fig, Path(figures_dir) / "dataset_class_distribution.png")


def _bar_metric_plot(
    df: pd.DataFrame,
    title: str,
    fname: str,
    figures_dir: Path,
    metrics: tuple[str, ...] = ("macro_f1", "accuracy"),
) -> Path | None:
    if df.empty:
        return None
    agg = df.groupby("model")[list(metrics)].agg(["mean", "std"]).reset_index()
    models = agg["model"].tolist()
    x = np.arange(len(models))
    width = 0.35 if len(metrics) == 2 else 0.55
    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(models) + 2.5), 4.6))
    palette = ["#1a365d", "#3182ce", "#38a169"]
    for idx, m in enumerate(metrics):
        means = agg[(m, "mean")].to_numpy(dtype=float)
        stds = np.nan_to_num(agg[(m, "std")].to_numpy(dtype=float), nan=0.0)
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        bars = ax.bar(x + offset, means, width=width, yerr=stds, capsize=3,
                      label=m.replace("_", " ").title(), color=palette[idx % len(palette)], alpha=0.9)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=28, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _save_fig(fig, figures_dir / fname)


def plot_model_comparisons(runs_df: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Generate family-specific, track-specific, and computational comparison figures."""
    if runs_df.empty:
        return []
    fdir = Path(figures_dir)
    saved: list[Path] = []

    fundus_df = runs_df[runs_df["track"] == "fundus_3class"]
    if not fundus_df.empty:
        for cat, fname, title in [
            ("cnn", "cnn_comparison.png", "Fundus CNN Models Comparison"),
            ("rnn", "rnn_comparison.png", "Fundus Standalone RNN Models Comparison"),
            ("hybrid", "hybrid_comparison.png", "Fundus Hybrid CNN-RNN Models Comparison"),
        ]:
            sub = fundus_df[fundus_df["category"] == cat]
            p = _bar_metric_plot(sub, title, fname, fdir)
            if p:
                saved.append(p)
        p = _bar_metric_plot(fundus_df, "Overall Kim Fundus Architecture Comparison", "overall_fundus_comparison.png", fdir)
        if p:
            saved.append(p)

    for track, fname, title in [
        ("oct_individual", "oct_individual_comparison.png", "OCT Individual B-Scan Classification"),
        ("oct_sequence", "oct_sequence_comparison.png", "OCT Ordered Sequence Classification"),
        ("oct_severity", "severity_comparison.png", "OCT Exploratory Severity Grading (G1-G4)"),
    ]:
        sub = runs_df[runs_df["track"] == track]
        p = _bar_metric_plot(sub, title, fname, fdir)
        if p:
            saved.append(p)

    # Computational plots (using primary fundus track if present, else all aggregated by model)
    comp_df = fundus_df if not fundus_df.empty else runs_df
    for col, fname, title, ylabel in [
        ("total_params", "parameter_comparison.png", "Model Parameter Count Comparison", "Parameters (Millions)"),
        ("train_duration_sec", "training_time_comparison.png", "Training Duration Comparison", "Seconds"),
        ("inference_latency_ms", "inference_time_comparison.png", "Inference Latency Comparison", "ms / sample"),
    ]:
        if col not in comp_df.columns:
            continue
        agg = comp_df.groupby("model")[col].mean().sort_values(ascending=True)
        vals = agg.values / 1e6 if col == "total_params" else agg.values
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        bars = ax.barh(agg.index, vals, color="#4a5568", alpha=0.88)
        for b in bars:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2.0,
                    f" {b.get_width():.2f}", va="center", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=0.25)
        saved.append(_save_fig(fig, fdir / fname))
    return saved


def plot_per_class_performance(
    per_class_rows: list[dict[str, Any]],
    figures_dir: str | Path,
    filename: str = "per_class_performance.png",
    title: str = "Per-Class F1 Performance by Model",
) -> Path | None:
    """Plot per-class F1 scores grouped by model."""
    if not per_class_rows:
        return None
    df = pd.DataFrame(per_class_rows)
    piv = df.pivot_table(index="model", columns="class_name", values="f1", aggfunc="mean")
    if piv.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(piv) + 2.0), 4.6))
    piv.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=28, labelsize=8)
    ax.legend(title="Class", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _save_fig(fig, Path(figures_dir) / filename)

