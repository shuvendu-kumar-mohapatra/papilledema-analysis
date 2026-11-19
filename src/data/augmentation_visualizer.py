"""Visualization utilities for offline augmented datasets."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import resolve_path


def plot_augmentation_class_distributions(
    summary_df: pd.DataFrame,
    save_path: str | Path | None = "results/augmentation/augmentation_summary/augmentation_class_distributions.png",
) -> plt.Figure:
    """Plot before vs after class distribution bar charts for each track."""
    datasets = summary_df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    colors_orig = "#1f77b4"
    colors_aug = "#2ca02c"

    for idx, dname in enumerate(datasets):
        ax = axes[0, idx]
        sub = summary_df[summary_df["dataset"] == dname].sort_values("class_label")
        x = np.arange(len(sub))
        width = 0.35

        ax.bar(x - width / 2, sub["original_count"], width, label="Original", color=colors_orig, alpha=0.85)
        ax.bar(x + width / 2, sub["final_total"], width, label="Augmented", color=colors_aug, alpha=0.85)

        ax.set_title(dname.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["class_label"], fontsize=10)
        ax.set_ylabel("Sample Count", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True)

    plt.tight_layout()
    if save_path:
        p = resolve_path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        pdf_path = p.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    return fig


def create_side_by_side_augmentation_grid(
    manifest_df: pd.DataFrame,
    track: str,
    n_examples: int = 4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a grid showing original images alongside their synthetic counterparts."""
    sub = manifest_df[(manifest_df["dataset"] == track) & (manifest_df["is_synthetic"])].copy()
    if sub.empty:
        raise ValueError(f"No synthetic samples found for track '{track}'")

    unique_parents = sub["original_image_id"].drop_duplicates().head(n_examples).tolist()
    fig, axes = plt.subplots(len(unique_parents), 3, figsize=(9, 3 * len(unique_parents)), squeeze=False)

    for r_idx, parent_id in enumerate(unique_parents):
        orig_row = manifest_df[manifest_df["image_id"] == parent_id].iloc[0]
        syn_rows = manifest_df[manifest_df["original_image_id"] == parent_id]
        syn_rows = syn_rows[syn_rows["is_synthetic"]].head(2)

        # Plot original
        orig_img = Image.open(resolve_path(orig_row["output_path"]))
        ax_orig = axes[r_idx, 0]
        ax_orig.imshow(orig_img, cmap="gray" if "oct" in track else None)
        ax_orig.set_title(f"Original: {parent_id[:16]}\n({orig_row['class_label']})", fontsize=9)
        ax_orig.axis("off")

        # Plot synthetic counterparts
        for c_idx, (_, syn_row) in enumerate(syn_rows.iterrows()):
            ax_syn = axes[r_idx, c_idx + 1]
            syn_img = Image.open(resolve_path(syn_row["output_path"]))
            ax_syn.imshow(syn_img, cmap="gray" if "oct" in track else None)
            ax_syn.set_title(f"Aug {c_idx+1}: {syn_row['image_id'][:16]}\n{syn_row['augmentation_type']}", fontsize=8)
            ax_syn.axis("off")

    plt.tight_layout()
    if save_path:
        p = resolve_path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        pdf_path = p.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    return fig
