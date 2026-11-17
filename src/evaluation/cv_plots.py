"""Publication-ready PNG & PDF plots for Phase 5 comprehensive CV model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.cross_validation.runner import CV_TRACK_SPECS
from src.utils.config import resolve_path

FAMILY_COLORS = {
    "cnn": "#2b5c8f",
    "hybrid": "#1b9e77",
    "rnn": "#d95f02",
    "baseline": "#7570b3",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.titlesize": 12.5,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def _save_dual(fig: plt.Figure, fig_dir: Path, pub_dir: Path, stem_name: str) -> list[Path]:
    png_p = fig_dir / f"{stem_name}.png"
    pdf_p = fig_dir / f"{stem_name}.pdf"
    fig.savefig(png_p, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_p, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(png_p, pub_dir / png_p.name)
    shutil.copy2(pdf_p, pub_dir / pdf_p.name)
    return [png_p, pdf_p]


def _plot_track_suite(
    track: str,
    runs_t: pd.DataFrame,
    fold_t: pd.DataFrame,
    sum_t: pd.DataFrame,
    pcls_t: pd.DataFrame,
    oof_t: pd.DataFrame,
    oof_preds_store: dict[tuple[str, str], pd.DataFrame],
    cv_raw: dict[str, Any],
    fig_dir: Path,
    pub_dir: Path,
) -> list[Path]:
    _, _, class_names, _ = CV_TRACK_SPECS[track]
    out_paths: list[Path] = []
    sum_ord = sum_t.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    models = sum_ord["model"].tolist()
    cols = [FAMILY_COLORS.get(c, "#555555") for c in sum_ord["category"]]
    ckpt_root = resolve_path(cv_raw.get("cross_validation", cv_raw)["paths"]["checkpoints_dir"])

    # 1. Overall Model Comparison (Macro-F1, Balanced Acc, Accuracy, MCC, Cohen's Kappa)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, max(4.2, 0.45 * len(models) + 1.5)))
    y_pos = np.arange(len(models))
    axes[0].barh(y_pos, sum_ord["macro_f1_mean"], xerr=sum_ord["macro_f1_std"], capsize=3, color=cols, edgecolor="#222222", alpha=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.08)
    axes[0].set_xlabel("5-Fold Mean Macro-F1 (± SD)")
    axes[0].set_title(f"{track}: Model Ranking by 5-Fold Macro-F1")
    axes[0].grid(axis="x", alpha=0.25, linestyle="--")
    for idx, (m_val, s_val) in enumerate(zip(sum_ord["macro_f1_mean"], sum_ord["macro_f1_std"])):
        axes[0].text(min(1.01, m_val + s_val + 0.012), idx, f"{m_val:.3f}±{s_val:.3f}", va="center", fontsize=8)

    w = 0.25
    x = np.arange(len(models))
    axes[1].bar(x - w, sum_ord["accuracy_mean"], w, label="Accuracy", color="#4e79a7", edgecolor="#222")
    axes[1].bar(x, sum_ord["balanced_accuracy_mean"], w, label="Balanced Acc", color="#59a14f", edgecolor="#222")
    axes[1].bar(x + w, sum_ord["mcc_mean"], w, label="MCC", color="#e15759", edgecolor="#222")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=32, ha="right", fontsize=8)
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{track}: Accuracy, Balanced Accuracy & MCC")
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    fig.suptitle(f"5-Fold CV Performance Overview — {track}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_01_model_comparison"))

    # 2. Fold-wise Stability (Macro-F1 & Accuracy across Folds 1..5)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    cmap = plt.get_cmap("tab10")
    for idx, m_name in enumerate(models):
        sub_f = fold_t[fold_t["model"] == m_name].sort_values("fold")
        axes[0].plot(sub_f["fold"], sub_f["macro_f1"], marker="o", linewidth=1.6, color=cmap(idx % 10), label=m_name)
        axes[1].plot(sub_f["fold"], sub_f["accuracy"], marker="s", linewidth=1.6, color=cmap(idx % 10), label=m_name)
    for ax, ttl, ylab in ((axes[0], "Fold-Wise Macro-F1", "Macro-F1"), (axes[1], "Fold-Wise Accuracy", "Accuracy")):
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Outer Test Fold")
        ax.set_ylabel(ylab)
        ax.set_title(f"{track}: {ttl}")
        ax.set_ylim(-0.02, 1.04)
        ax.grid(alpha=0.25, linestyle="--")
    axes[1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.suptitle(f"Fold-Wise CV Stability Across All Models — {track}")
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_02_foldwise_stability"))

    # 3. Per-Class Precision, Recall, F1 (OOF)
    oof_pcls = pcls_t[pcls_t["scope"] == "oof"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=True)
    for ax, metric_col, title in (
        (axes[0], "precision", "Per-Class Precision (OOF)"),
        (axes[1], "recall", "Per-Class Recall / Sensitivity (OOF)"),
        (axes[2], "f1", "Per-Class F1-Score (OOF)"),
    ):
        piv = oof_pcls.pivot(index="model", columns="class", values=metric_col).reindex(index=models, columns=class_names)
        piv.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="#222222", width=0.78)
        ax.set_title(title)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Score")
    fig.suptitle(f"Class-Wise Precision, Recall, and F1 Across All Models — {track}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_03_per_class_metrics"))

    # 4. Metric Heatmaps (Model Metrics + Class-wise F1)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, max(4.2, 0.45 * len(models) + 1.5)), gridspec_kw={"width_ratios": [2.1, 1.0]})
    m_cols = ["macro_f1_mean", "accuracy_mean", "balanced_accuracy_mean", "macro_precision_mean", "macro_recall_mean", "weighted_f1_mean", "mcc_mean", "cohen_kappa_mean", "roc_auc_ovr_mean", "ece_mean"]
    hm1 = sum_ord.set_index("model")[m_cols]
    hm1.columns = [c.replace("_mean", "").replace("_ovr", "") for c in hm1.columns]
    sns.heatmap(hm1, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], cbar_kws={"shrink": 0.8}, linewidths=0.4)
    axes[0].set_title(f"{track}: Model-Wise Metric Heatmap (5-Fold Mean)")
    axes[0].tick_params(axis="x", rotation=32)

    hm2 = oof_pcls.pivot(index="model", columns="class", values="f1").reindex(index=models, columns=class_names)
    sns.heatmap(hm2, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, ax=axes[1], cbar_kws={"shrink": 0.8}, linewidths=0.4)
    axes[1].set_title(f"{track}: Class-Wise OOF F1 Heatmap")
    axes[1].set_ylabel("")
    fig.tight_layout()
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_04_metric_heatmaps"))

    # 5. Confusion Matrices for EVERY model (OOF pooled)
    n_m = len(models)
    ncols = 5 if n_m > 5 else n_m
    nrows = int(np.ceil(n_m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows))
    axes_flat = np.asarray(axes).ravel()
    for i, m_name in enumerate(models):
        ax = axes_flat[i]
        row_oof = oof_t[oof_t["model"] == m_name].iloc[0]
        cm = np.asarray(json.loads(row_oof["oof_confusion_matrix_json"]), dtype=np.int64)
        cm_norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
        sns.heatmap(cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, xticklabels=class_names, yticklabels=class_names)
        for r_i in range(cm.shape[0]):
            for c_i in range(cm.shape[1]):
                val = cm[r_i, c_i]
                pct = 100.0 * cm_norm[r_i, c_i]
                txt_col = "white" if cm_norm[r_i, c_i] > 0.55 else "#111111"
                ax.text(c_i + 0.5, r_i + 0.5, f"{val}\n({pct:.0f}%)", ha="center", va="center", fontsize=7.5, color=txt_col, fontweight="bold")
        ax.set_title(f"{m_name}\nOOF F1={row_oof['oof_macro_f1']:.3f}", fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=7.5)
        ax.tick_params(axis="y", rotation=0, labelsize=7.5)
        if i % ncols == 0:
            ax.set_ylabel("True")
        ax.set_xlabel("Pred")
    for j in range(n_m, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(f"Pooled 5-Fold Out-of-Fold Confusion Matrices for All Models — {track}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_05_confusion_matrices_all"))

    # 6. Macro-Averaged ROC, Precision-Recall, and Reliability/Calibration Curves (OOF)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    prob_cols = [f"prob_{c}" for c in class_names]
    grid_fpr = np.linspace(0.0, 1.0, 101)
    grid_rec = np.linspace(0.0, 1.0, 101)
    for idx, m_name in enumerate(models):
        pdf = oof_preds_store[(track, m_name)]
        yt = pdf["true_idx"].to_numpy(dtype=np.int64)
        probs = pdf[prob_cols].to_numpy(dtype=np.float64)
        col = cmap(idx % 10)
        # Macro OvR ROC
        tprs, precs = [], []
        for c_i in range(len(class_names)):
            y_bin = (yt == c_i).astype(int)
            if 0 < y_bin.sum() < len(y_bin):
                fpr, tpr, _ = roc_curve(y_bin, probs[:, c_i])
                tprs.append(np.interp(grid_fpr, fpr, tpr))
                p_c, r_c, _ = precision_recall_curve(y_bin, probs[:, c_i])
                precs.append(np.interp(grid_rec, r_c[::-1], p_c[::-1]))
        if tprs:
            axes[0].plot(grid_fpr, np.mean(tprs, axis=0), color=col, linewidth=1.6, label=m_name)
        if precs:
            axes[1].plot(grid_rec, np.mean(precs, axis=0), color=col, linewidth=1.6, label=m_name)
        # Confidence calibration curve (top-1 accuracy vs predicted confidence)
        conf = np.max(probs, axis=1)
        corr = (np.argmax(probs, axis=1) == yt).astype(int)
        frac_pos, mean_pred = calibration_curve(corr, conf, n_bins=8, strategy="uniform")
        axes[2].plot(mean_pred, frac_pos, marker="o", linewidth=1.4, markersize=4, color=col, label=m_name)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title(f"{track}: Macro OvR ROC Curves (OOF)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{track}: Macro OvR PR Curves (OOF)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calib.")
    axes[2].set_title(f"{track}: Top-Class Reliability Curve (OOF)")
    axes[2].set_xlabel("Mean Predicted Confidence")
    axes[2].set_ylabel("Observed Empirical Accuracy")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(alpha=0.25, linestyle="--")
    axes[2].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7.5)
    fig.suptitle(f"Discrimination & Calibration Curves (Out-of-Fold Pooled) — {track}")
    fig.tight_layout(rect=(0, 0, 0.89, 0.94))
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_06_roc_pr_calibration"))

    # 7. Training & Validation Loss / Macro-F1 Learning Curves (Mean +/- SD across 5 folds)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2))
    metrics_grid = [
        (axes[0, 0], "train_loss", "Training Loss (5-Fold Mean)", "Loss"),
        (axes[0, 1], "val_loss", "Validation Loss (5-Fold Mean)", "Loss"),
        (axes[1, 0], "train_macro_f1", "Training Macro-F1 (5-Fold Mean)", "Macro-F1"),
        (axes[1, 1], "val_macro_f1", "Validation Macro-F1 (5-Fold Mean)", "Macro-F1"),
    ]
    for idx, m_name in enumerate(models):
        h_list = []
        sub_r = runs_t[runs_t["model"] == m_name]
        for cv_id in sub_r["cv_exp_id"]:
            hp = ckpt_root / cv_id / "history.csv"
            if hp.exists():
                h_list.append(pd.read_csv(hp))
        if not h_list:
            continue
        cat_h = pd.concat(h_list, ignore_index=True)
        agg = cat_h.groupby("epoch")[["train_loss", "val_loss", "train_macro_f1", "val_macro_f1"]].agg(["mean", "std"]).reset_index()
        col = cmap(idx % 10)
        ep = agg["epoch"].to_numpy()
        for ax, col_name, _, _ in metrics_grid:
            mu = agg[col_name]["mean"].to_numpy()
            sd = agg[col_name]["std"].fillna(0).to_numpy()
            ax.plot(ep, mu, color=col, linewidth=1.6, label=m_name)
            ax.fill_between(ep, mu - sd, mu + sd, color=col, alpha=0.09)
    for ax, _, ttl, ylab in metrics_grid:
        ax.set_title(f"{track}: {ttl}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, linestyle="--")
    axes[0, 1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.suptitle(f"5-Fold Learning Curves (Mean ± SD Across Folds) — {track}")
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_07_learning_curves"))

    # 8. Computational Cost Suite (Train Time, Latency, Params, Ckpt Size, Peak GPU Mem, Perf vs Cost)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    panels = [
        (axes[0, 0], "train_time_mean_sec", "Mean Fold Train Time (s)", "Seconds"),
        (axes[0, 1], "inference_latency_ms", "Inference Latency (ms/sample)", "ms / sample"),
        (axes[0, 2], "total_params", "Total Parameters (Millions)", "Parameters (M)", 1e-6),
        (axes[1, 0], "ckpt_size_mb", "Checkpoint Disk Size (MB)", "MB"),
        (axes[1, 1], "peak_gpu_mem_mb_max", "Max Peak GPU Memory (MB)", "MB VRAM"),
    ]
    for tup in panels:
        ax, col_key, ttl, ylab = tup[0], tup[1], tup[2], tup[3]
        scale = tup[4] if len(tup) > 4 else 1.0
        vals = sum_ord[col_key].to_numpy() * scale
        ax.bar(np.arange(len(models)), vals, color=cols, edgecolor="#222")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(ttl)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Performance vs Cost Pareto scatter
    ax_p = axes[1, 2]
    for r in sum_ord.itertuples(index=False):
        c_fam = FAMILY_COLORS.get(r.category, "#555")
        ax_p.scatter(r.inference_latency_ms, r.macro_f1_mean, s=max(40, np.sqrt(r.total_params) * 0.045), color=c_fam, edgecolor="#111", alpha=0.85)
        ax_p.annotate(r.model, (r.inference_latency_ms, r.macro_f1_mean), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax_p.set_xlabel("Inference Latency (ms / sample)")
    ax_p.set_ylabel("5-Fold Macro-F1")
    ax_p.set_title("Macro-F1 vs. Latency (Size ~ √Params)")
    ax_p.grid(alpha=0.25, linestyle="--")
    fig.suptitle(f"Computational Footprint & Performance-Cost Tradeoff — {track}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_paths.extend(_save_dual(fig, fig_dir, pub_dir, f"fig_{track}_08_computational_and_cost"))

    return out_paths


def generate_all_cv_evaluation_figures(
    runs_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    per_cls_df: pd.DataFrame,
    oof_df_all: pd.DataFrame,
    oof_preds_store: dict[tuple[str, str], pd.DataFrame],
    seq_lvl_df: pd.DataFrame,
    cv_raw: dict[str, Any],
    fig_dir: Path,
    pub_dir: Path,
) -> list[Path]:
    """Generate all per-track and cross-track PNG+PDF figures for Phase 5."""
    _apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []

    for track in sorted(runs_df["track"].unique()):
        all_paths.extend(_plot_track_suite(
            track=track,
            runs_t=runs_df[runs_df["track"] == track],
            fold_t=fold_df[fold_df["track"] == track],
            sum_t=model_summary_df[model_summary_df["track"] == track],
            pcls_t=per_cls_df[per_cls_df["track"] == track],
            oof_t=oof_df_all[oof_df_all["track"] == track],
            oof_preds_store=oof_preds_store,
            cv_raw=cv_raw,
            fig_dir=fig_dir,
            pub_dir=pub_dir,
        ))

    # Dedicated Track 3 Window-Level (N=146) vs Sequence-Level (N=21) Comparison
    if not seq_lvl_df.empty:
        o3_oof = oof_df_all[oof_df_all["track"] == "oct_sequence"].merge(seq_lvl_df, on=["track", "model", "category"])
        o3_oof = o3_oof.sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
        x = np.arange(len(o3_oof))
        w = 0.36
        b1 = axes[0].bar(x - w / 2, o3_oof["oof_macro_f1"], w, label="Window-Level OOF (N=146)", color="#2b5c8f", edgecolor="#222")
        b2 = axes[0].bar(x + w / 2, o3_oof["seq_macro_f1"], w, label="Sequence-Level OOF (N=21)", color="#1b9e77", edgecolor="#222")
        for bars in (b1, b2):
            for b in bars:
                axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(o3_oof["model"], rotation=25, ha="right")
        axes[0].set_ylim(0, 1.12)
        axes[0].set_ylabel("Macro-F1")
        axes[0].set_title("OCT Sequence: Window-Level vs. Sequence-Level Macro-F1")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.25, linestyle="--")

        best_seq = o3_oof.iloc[0]
        cm_s = np.asarray(json.loads(best_seq["confusion_matrix_json"]), dtype=np.int64)
        sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens", ax=axes[1], xticklabels=CV_TRACK_SPECS["oct_sequence"][2], yticklabels=CV_TRACK_SPECS["oct_sequence"][2])
        axes[1].set_title(f"Best Model ({best_seq['model']}) Sequence-Level CM (N={int(best_seq['n_sequences'])})")
        axes[1].set_xlabel("Predicted Sequence Label")
        axes[1].set_ylabel("True Sequence Label")
        fig.suptitle("EXP-03 Separate Granularity Evaluation (Windows Never Mixed With Sequences)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        all_paths.extend(_save_dual(fig, fig_dir, pub_dir, "fig_oct_sequence_09_window_vs_sequence"))

    # Cross-Track 1: Architecture Heatmap + Family Comparison + CV Stability
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    piv_f1 = model_summary_df.pivot(index="model", columns="track", values="macro_f1_mean")
    track_cols = [t for t in ("fundus_3class", "oct_individual", "oct_sequence", "oct_severity") if t in piv_f1.columns]
    piv_f1 = piv_f1[track_cols].sort_values(track_cols[0], ascending=False)
    sns.heatmap(piv_f1, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=axes[0], cbar_kws={"shrink": 0.8}, linewidths=0.4)
    axes[0].set_title("Cross-Track 5-Fold Mean Macro-F1\n(Separate Tasks; Blank = CNN/MLP Not Applicable)")
    axes[0].tick_params(axis="x", rotation=22)

    fam = model_summary_df.groupby(["track", "category"])["macro_f1_mean"].mean().unstack("category").reindex(track_cols)
    fam.plot(kind="bar", ax=axes[1], color=[FAMILY_COLORS.get(c, "#777") for c in fam.columns], edgecolor="#222")
    axes[1].set_title("Architecture Family Mean Macro-F1 per Track")
    axes[1].set_ylabel("Mean 5-Fold Macro-F1")
    axes[1].set_ylim(0, 1.08)
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    piv_sd = model_summary_df.pivot(index="model", columns="track", values="macro_f1_std").reindex(index=piv_f1.index, columns=track_cols)
    sns.heatmap(piv_sd, annot=True, fmt=".3f", cmap="OrRd", ax=axes[2], cbar_kws={"shrink": 0.8}, linewidths=0.4)
    axes[2].set_title("Cross-Validation Instability (5-Fold SD Macro-F1)")
    axes[2].tick_params(axis="x", rotation=22)
    axes[2].set_ylabel("")
    fig.suptitle("Cross-Track Architecture Family & Cross-Validation Stability Synthesis")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    all_paths.extend(_save_dual(fig, fig_dir, pub_dir, "fig_cross_01_architecture_heatmaps_and_families"))

    # Cross-Track 2: Complexity & Runtime Pareto per Track (not averaged across tasks)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    trk_markers = {"fundus_3class": "o", "oct_individual": "s", "oct_sequence": "^", "oct_severity": "D"}
    for trk, g in model_summary_df.groupby("track"):
        mk = trk_markers.get(trk, "o")
        pos_params = np.maximum(g["total_params"] / 1e6, 0.01)
        axes[0].scatter(pos_params, g["macro_f1_mean"], marker=mk, s=75, alpha=0.85, label=trk)
        axes[1].scatter(np.maximum(g["train_time_total_sec"] / 60.0, 0.001), g["macro_f1_mean"], marker=mk, s=75, alpha=0.85, label=trk)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Model Parameters (Millions, log scale)")
    axes[0].set_ylabel("5-Fold Mean Macro-F1")
    axes[0].set_title("Macro-F1 vs. Model Complexity Across Tracks")
    axes[1].set_xlabel("Total 5-Fold Training Time (Minutes)")
    axes[1].set_ylabel("5-Fold Mean Macro-F1")
    axes[1].set_title("Macro-F1 vs. 5-Fold Training Time Across Tracks")
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=True)
    fig.suptitle("Performance vs. Complexity and Compute Time (Tracks Plotted Independently)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    all_paths.extend(_save_dual(fig, fig_dir, pub_dir, "fig_cross_02_complexity_and_runtime_pareto"))

    return all_paths

