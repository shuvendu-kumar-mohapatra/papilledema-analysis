"""Publication-quality training curves, confusion matrices, ROC, and PR curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(
    histories_by_label: dict[str, pd.DataFrame],
    figures_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Generate training loss, validation loss, accuracy, and macro F1 curve figures."""
    if not histories_by_label:
        return []
    fdir = Path(figures_dir)
    saved: list[Path] = []
    pre = f"{prefix}_" if prefix else ""
    specs = [
        ("train_loss", f"Training Loss Curves ({prefix})" if prefix else "Training Loss Curves", "Loss", f"{pre}training_loss_curves.png", False),
        ("val_loss", f"Validation Loss Curves ({prefix})" if prefix else "Validation Loss Curves", "Validation Loss", f"{pre}validation_loss_curves.png", False),
        ("val_accuracy", f"Accuracy Curves — Train Dashed / Val Solid ({prefix})" if prefix else "Accuracy Curves (Train Dashed / Val Solid)", "Accuracy", f"{pre}accuracy_curves.png", True),
        ("val_macro_f1", f"Macro F1 Curves — Train Dashed / Val Solid ({prefix})" if prefix else "Macro F1 Curves (Train Dashed / Val Solid)", "Macro F1", f"{pre}macro_f1_curves.png", True),
    ]
    for col, title, ylabel, fname, pair_train in specs:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        plotted = False
        for label, hdf in sorted(histories_by_label.items()):
            if hdf.empty or col not in hdf.columns:
                continue
            line, = ax.plot(hdf["epoch"], hdf[col], linewidth=1.8, label=label)
            if pair_train:
                tr_col = col.replace("val_", "train_")
                if tr_col in hdf.columns:
                    ax.plot(hdf["epoch"], hdf[tr_col], linestyle="--", alpha=0.55, color=line.get_color())
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        saved.append(_save_fig(fig, fdir / fname))
    return saved


def plot_confusion_matrices(
    eval_items: list[dict[str, Any]],
    figures_dir: str | Path,
    filename: str = "confusion_matrices.png",
) -> Path | None:
    """Plot confusion matrices grid across evaluated models."""
    valid = [e for e in eval_items if e.get("confusion_matrix")]
    if not valid:
        return None
    n = len(valid)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    for idx, item in enumerate(valid):
        ax = axes_arr[idx]
        cm = np.asarray(item["confusion_matrix"], dtype=float)
        classes = item["class_names"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(item["title"], fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")
    return _save_fig(fig, Path(figures_dir) / filename)


def plot_roc_and_pr_curves(
    pred_dfs_by_label: dict[str, pd.DataFrame],
    class_names: list[str],
    figures_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Plot macro-averaged One-vs-Rest ROC and Precision-Recall curves per model."""
    if not pred_dfs_by_label:
        return []
    fdir = Path(figures_dir)
    pre = f"{prefix}_" if prefix else ""
    saved: list[Path] = []
    fig_roc, ax_roc = plt.subplots(figsize=(7.5, 5.5))
    fig_pr, ax_pr = plt.subplots(figsize=(7.5, 5.5))

    grid = np.linspace(0.0, 1.0, 200)
    plotted = False
    for label, pdf in sorted(pred_dfs_by_label.items()):
        prob_cols = [f"prob_{c}" for c in class_names]
        if pdf.empty or any(c not in pdf.columns for c in prob_cols):
            continue
        yt = pdf["true_idx"].to_numpy()
        probs = pdf[prob_cols].to_numpy(dtype=float)

        tprs, aucs = [], []
        precs, pr_aucs = [], []
        for ci in range(len(class_names)):
            yb = (yt == ci).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            fpr, tpr, _ = roc_curve(yb, probs[:, ci])
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(auc(fpr, tpr))

            prec, rec, _ = precision_recall_curve(yb, probs[:, ci])
            precs.append(np.interp(grid, rec[::-1], prec[::-1]))
            pr_aucs.append(float(average_precision_score(yb, probs[:, ci])))

        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
            ax_roc.plot(grid, mean_tpr, linewidth=1.8, label=f"{label} (macro AUC={np.mean(aucs):.3f})")
            mean_prec = np.mean(precs, axis=0)
            ax_pr.plot(grid, mean_prec, linewidth=1.8, label=f"{label} (macro PR-AUC={np.mean(pr_aucs):.3f})")
            plotted = True

    if not plotted:
        plt.close(fig_roc)
        plt.close(fig_pr)
        return []

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_title(f"Macro-Averaged OvR ROC Curves ({prefix})" if prefix else "Macro-Averaged OvR ROC Curves", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(alpha=0.25)
    ax_roc.legend(fontsize=8)
    saved.append(_save_fig(fig_roc, fdir / f"{pre}roc_curves.png"))

    ax_pr.set_title(f"Macro-Averaged Precision-Recall Curves ({prefix})" if prefix else "Macro-Averaged Precision-Recall Curves", fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(fontsize=8)
    saved.append(_save_fig(fig_pr, fdir / f"{pre}pr_curves.png"))
    return saved

