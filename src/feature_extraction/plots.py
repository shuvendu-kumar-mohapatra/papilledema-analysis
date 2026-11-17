"""Publication-ready PNG & PDF plots for Phase 3 handcrafted and deep feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES


def _apply_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def _save_pair(fig: plt.Figure, stem: Path) -> list[Path]:
    png_p = stem.with_suffix(".png")
    pdf_p = stem.with_suffix(".pdf")
    fig.savefig(png_p, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_p, bbox_inches="tight")
    plt.close(fig)
    return [png_p, pdf_p]


def plot_handcrafted_distributions(f_hc_df: pd.DataFrame, o_hc_df: pd.DataFrame, imp_tables: dict[str, pd.DataFrame], fig_dir: Path) -> list[Path]:
    """Boxplots of top discriminative interpretable handcrafted features across classes."""
    f_top = [f for f in imp_tables["fundus_3class"]["feature"].tolist() if "_hist_" not in f and "_bin_" not in f][:4]
    o_top = [f for f in imp_tables["oct_individual"]["feature"].tolist() if "_hist_" not in f and "_bin_" not in f][:4]

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    pal_3 = ["#2b5c8f", "#d95f02", "#1b9e77"]
    for c_idx, feat in enumerate(f_top):
        ax = axes[0, c_idx]
        data = [f_hc_df.loc[f_hc_df["label"] == cls, feat].to_numpy() for cls in FUNDUS_CLASSES]
        bp = ax.boxplot(data, tick_labels=["Normal", "Pap.", "Pseudo."], patch_artist=True)
        for patch, col in zip(bp["boxes"], pal_3):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_title(f"Fundus: {feat}", fontsize=9.5)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    for c_idx, feat in enumerate(o_top):
        ax = axes[1, c_idx]
        data = [o_hc_df.loc[o_hc_df["label"] == cls, feat].to_numpy() for cls in OCT_3CLASSES]
        bp = ax.boxplot(data, tick_labels=["Normal", "Pap.", "Pseudo."], patch_artist=True)
        for patch, col in zip(bp["boxes"], pal_3):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_title(f"OCT: {feat}", fontsize=9.5)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Top Class-Discriminative Handcrafted Image Features (Fundus & Individual OCT)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_pair(fig, fig_dir / "fig01_handcrafted_feature_distributions")


def plot_correlation_heatmaps(f_corr: pd.DataFrame, o_corr: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Side-by-side Pearson correlation heatmaps for interpretable handcrafted feature subsets."""
    f_sub = f_corr.iloc[:18, :18]
    o_sub = o_corr.iloc[:18, :18]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
    sns.heatmap(f_sub, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, cbar_kws={"shrink": 0.75}, ax=axes[0])
    axes[0].set_title("Fundus Handcrafted Feature Correlation (Intensity/RGB/GLCM/Edges)")
    axes[0].tick_params(axis="x", rotation=55, labelsize=7.5)
    axes[0].tick_params(axis="y", labelsize=7.5)

    sns.heatmap(o_sub, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, cbar_kws={"shrink": 0.75}, ax=axes[1])
    axes[1].set_title("OCT Handcrafted Feature Correlation (Intensity/GLCM/Edges)")
    axes[1].tick_params(axis="x", rotation=55, labelsize=7.5)
    axes[1].tick_params(axis="y", labelsize=7.5)

    fig.suptitle("Handcrafted Feature Correlation Structure (Fundus vs. OCT Kept Strictly Separate)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_pair(fig, fig_dir / "fig02_handcrafted_correlation_heatmap")


def plot_pca_variance(pca_var_df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Cumulative explained variance curves up to 50 PCs across Handcrafted and CNN embeddings."""
    tracks = [("fundus_3class", "EXP-01: Fundus 3-Class"), ("oct_individual", "EXP-02: Individual OCT"), ("oct_severity", "EXP-04: OCT Severity")]
    space_styles = {
        "handcrafted": ("Handcrafted", "#333333", "--"),
        "resnet50": ("ResNet50 (2048-d)", "#2b5c8f", "-"),
        "densenet121": ("DenseNet121 (1024-d)", "#1b9e77", "-"),
        "efficientnet_b0": ("EfficientNet-B0 (1280-d)", "#d95f02", "-"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4))
    for ax, (trk, title) in zip(axes, tracks):
        sub = pca_var_df[pca_var_df["track"] == trk]
        for sp, (lbl, col, ls) in space_styles.items():
            s2 = sub[sub["feature_space"] == sp]
            if not s2.empty:
                ax.plot(s2["n_components"], s2["cum_explained_variance"] * 100.0, label=lbl, color=col, linestyle=ls, linewidth=1.8)
        ax.axhline(90.0, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative Explained Variance (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)
    fig.suptitle("PCA Cumulative Explained Variance: Handcrafted Features vs. Deep CNN Embeddings")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_pair(fig, fig_dir / "fig03_pca_scree_variance")


def plot_feature_space_2d(
    dr_df: pd.DataFrame,
    class_order: list[str],
    title: str,
    stem: Path,
) -> list[Path]:
    """4x3 exploratory grid (Handcrafted, ResNet50, DenseNet121, EfficientNet-B0 x PCA, t-SNE, UMAP)."""
    spaces = [
        ("handcrafted", "Handcrafted"),
        ("resnet50", "ResNet50 (2048-d)"),
        ("densenet121", "DenseNet121 (1024-d)"),
        ("efficientnet_b0", "EfficientNet-B0 (1280-d)"),
    ]
    methods = [("pca", "PCA (2D)"), ("tsne", "t-SNE (2D)"), ("umap", "UMAP (2D)")]
    palette = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]

    fig, axes = plt.subplots(4, 3, figsize=(13.8, 14.5))
    for r, (sp_key, sp_lbl) in enumerate(spaces):
        sub = dr_df[dr_df["feature_space"] == sp_key]
        for c, (m_key, m_lbl) in enumerate(methods):
            ax = axes[r, c]
            x_col, y_col = f"{m_key}_1", f"{m_key}_2"
            for cls_i, cls_name in enumerate(class_order):
                pts = sub[sub["label"] == cls_name]
                # Separate train/val vs held-out test markers transparently
                non_test = pts[pts["split"] != "test"]
                test_pts = pts[pts["split"] == "test"]
                if not non_test.empty:
                    ax.scatter(non_test[x_col], non_test[y_col], s=12, alpha=0.45, color=palette[cls_i], label=cls_name if (r == 0 and c == 0) else None)
                if not test_pts.empty:
                    ax.scatter(test_pts[x_col], test_pts[y_col], s=22, alpha=0.95, marker="^", edgecolor="#111111", linewidths=0.4, color=palette[cls_i])
            if r == 0:
                ax.set_title(m_lbl)
            if c == 0:
                ax.set_ylabel(f"{sp_lbl}\nDim 2", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(class_order), bbox_to_anchor=(0.5, 0.965), frameon=True)
    fig.suptitle(f"{title}\n(Exploratory Visualization: Circles=Train/Val, Triangles=Held-out Test; CNNs from seed-42 ckpts)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save_pair(fig, stem)


def plot_importance_and_baseline(imp_tables: dict[str, pd.DataFrame], clf_df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Top-12 Random Forest feature importances (fit strictly per CV training fold) + leak-free ML baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for ax, trk, title in (
        (axes[0], "fundus_3class", "Fundus Top-12 Handcrafted Features"),
        (axes[1], "oct_individual", "Individual OCT Top-12 Features"),
    ):
        top = imp_tables[trk].head(12).iloc[::-1]
        ax.barh(top["feature"], top["rf_mean_importance"], color="#2b5c8f" if "fundus" in trk else "#d95f02", edgecolor="#222222")
        ax.set_xlabel("Mean 5-Fold RF Gini Importance")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Rightmost: OOF Macro-F1 of handcrafted ML classifiers across all 3 tracks
    piv = clf_df.pivot(index="track", columns="classifier", values="oof_macro_f1").reindex(["fundus_3class", "oct_individual", "oct_severity"])
    x = np.arange(len(piv))
    w = 0.35
    b1 = axes[2].bar(x - w / 2, piv["LogisticRegression"], w, label="Logistic Regression", color="#4e79a7", edgecolor="#222222")
    b2 = axes[2].bar(x + w / 2, piv["RandomForest"], w, label="Random Forest", color="#59a14f", edgecolor="#222222")
    for bars in (b1, b2):
        for b in bars:
            axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Fundus\n(SGKFold)", "OCT 3-Class\n(Block CV)", "OCT Severity\n(Exploratory)"])
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Out-of-Fold Macro-F1 (Leak-Free)")
    axes[2].set_title("Handcrafted Feature ML Baseline (5-Fold OOF)")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Leak-Free Handcrafted Feature Importance and Baseline Separability (Fit Within Training Folds Only)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_pair(fig, fig_dir / "fig07_feature_importance_and_rf_baseline")


def generate_all_feature_figures(
    f_hc_df: pd.DataFrame,
    o_hc_df: pd.DataFrame,
    f_corr: pd.DataFrame,
    o_corr: pd.DataFrame,
    pca_var_df: pd.DataFrame,
    dr_tables: dict[str, pd.DataFrame],
    imp_tables: dict[str, pd.DataFrame],
    clf_df: pd.DataFrame,
    fig_dir: Path,
) -> list[Path]:
    _apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated.extend(plot_handcrafted_distributions(f_hc_df, o_hc_df, imp_tables, fig_dir))
    generated.extend(plot_correlation_heatmaps(f_corr, o_corr, fig_dir))
    generated.extend(plot_pca_variance(pca_var_df, fig_dir))
    generated.extend(plot_feature_space_2d(dr_tables["fundus_3class"], FUNDUS_CLASSES, "EXP-01 Fundus Feature Space Projections", fig_dir / "fig04_fundus_feature_space_2d"))
    generated.extend(plot_feature_space_2d(dr_tables["oct_individual"], OCT_3CLASSES, "EXP-02 Individual OCT Feature Space Projections", fig_dir / "fig05_oct_feature_space_2d"))
    generated.extend(plot_feature_space_2d(dr_tables["oct_severity"], OCT_SEVERITY_CLASSES, "EXP-04 OCT Severity Feature Space Projections", fig_dir / "fig06_oct_severity_feature_space_2d"))
    generated.extend(plot_importance_and_baseline(imp_tables, clf_df, fig_dir))
    return generated

