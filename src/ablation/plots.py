"""Publication-ready PNG & PDF plots for Phase 7 controlled ablation studies."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12.0,
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


def _plot_summary_grouped_bars(
    df: pd.DataFrame,
    models_order: list[str],
    colors: dict[str, str],
    title: str,
    subtitle: str,
    fig_dir: Path,
    pub_dir: Path,
    stem: str,
) -> list[Path]:
    if df.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, f"No rows available for {title}", ha="center", va="center")
        ax.axis("off")
        return _save_dual(fig, fig_dir, pub_dir, stem)

    tracks = sorted(df["track"].unique().tolist())
    models = [m for m in models_order if m in df["model"].unique()]
    if not models:
        models = sorted(df["model"].unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(max(11.5, 2.8 * len(tracks) + 4), 4.6), sharey=True)
    x = np.arange(len(tracks))
    width = 0.78 / max(1, len(models))

    for ax, (metric, err_col, xlabel) in zip(
        axes,
        [("macro_f1_mean", "macro_f1_std", "Macro-F1"), ("balanced_accuracy_mean", "balanced_accuracy_std", "Balanced Accuracy")],
    ):
        for i, m in enumerate(models):
            sub = df[df["model"] == m].set_index("track")
            vals = [float(sub.loc[t, metric]) if t in sub.index else 0.0 for t in tracks]
            errs = [float(sub.loc[t, err_col]) if (t in sub.index and err_col in sub.columns) else 0.0 for t in tracks]
            offset = (i - (len(models) - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width * 0.92,
                yerr=errs,
                capsize=3,
                color=colors.get(m, "#4c72b0"),
                alpha=0.92,
                label=m,
                edgecolor="white",
            )
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2.0, min(1.02, v + 0.025), f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90 if len(models) > 3 else 0)
        ax.set_xticks(x)
        ax.set_xticklabels(tracks, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.12)
        ax.set_ylabel(xlabel)
        ax.set_title(xlabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].legend(frameon=False, loc="lower left", ncol=min(len(models), 3))
    fig.suptitle(f"{title}\n{subtitle}", y=1.03)
    return _save_dual(fig, fig_dir, pub_dir, stem)


def _plot_abl4_trainability(abl4_df: pd.DataFrame, fig_dir: Path, pub_dir: Path) -> list[Path]:
    if abl4_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No Ablation 4 rows", ha="center", va="center")
        ax.axis("off")
        return _save_dual(fig, fig_dir, pub_dir, "abl04_backbone_trainability_comparison")

    modes = [m for m in ["frozen", "partial", "full"] if m in abl4_df["backbone_mode"].unique()]
    pairs = abl4_df[["track", "model"]].drop_duplicates().sort_values(["track", "model"]).to_records(index=False)
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, max(1, n_pairs), figsize=(6.2 * max(1, n_pairs), 4.6), sharey=True)
    if n_pairs == 1:
        axes = [axes]

    mode_colors = {"frozen": "#7f7f7f", "partial": "#e6ab02", "full": "#2b5c8f"}
    for ax, (trk, mdl) in zip(axes, pairs):
        sub = abl4_df[(abl4_df["track"] == trk) & (abl4_df["model"] == mdl)]
        agg = (
            sub.groupby("backbone_mode", as_index=False)
            .agg(
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                bal_acc_mean=("balanced_accuracy", "mean"),
                bal_acc_std=("balanced_accuracy", "std"),
                trainable_params=("trainable_params", "first"),
                total_params=("total_params", "first"),
                time_mean=("train_duration_sec", "mean"),
            )
            .fillna(0.0)
        )
        agg["backbone_mode"] = pd.Categorical(agg["backbone_mode"], categories=modes, ordered=True)
        agg = agg.sort_values("backbone_mode")

        x = np.arange(len(agg))
        w = 0.35
        b1 = ax.bar(
            x - w / 2,
            agg["macro_f1_mean"],
            width=w,
            yerr=agg["macro_f1_std"],
            capsize=3,
            color=[mode_colors.get(str(m), "#2b5c8f") for m in agg["backbone_mode"]],
            label="Macro-F1",
        )
        b2 = ax.bar(
            x + w / 2,
            agg["bal_acc_mean"],
            width=w,
            yerr=agg["bal_acc_std"],
            capsize=3,
            color="#1b9e77",
            alpha=0.82,
            label="Balanced Acc",
        )
        for rect, row in zip(b1, agg.itertuples(index=False)):
            pct = 100.0 * row.trainable_params / max(1, row.total_params)
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                min(1.02, row.macro_f1_mean + 0.02),
                f"{row.macro_f1_mean:.3f}\n({pct:.0f}% p)",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        for rect, row in zip(b2, agg.itertuples(index=False)):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                min(1.02, row.bal_acc_mean + 0.02),
                f"{row.bal_acc_mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.backbone_mode}\n({r.time_mean:.1f}s)" for r in agg.itertuples(index=False)])
        ax.set_ylim(0.0, 1.14)
        ax.set_title(f"{trk} — {mdl}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("5-Fold Mean Score (± SD)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Ablation 4: Controlled CNN Backbone Trainability (Frozen vs Partial vs Full)", y=1.03)
    return _save_dual(fig, fig_dir, pub_dir, "abl04_backbone_trainability_comparison")


def _plot_abl5_sequence_length(abl5_df: pd.DataFrame, abl5_lvl_df: pd.DataFrame, fig_dir: Path, pub_dir: Path) -> list[Path]:
    if abl5_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No Ablation 5 rows", ha="center", va="center")
        ax.axis("off")
        return _save_dual(fig, fig_dir, pub_dir, "abl05_oct_sequence_length_comparison")

    agg = (
        abl5_df.groupby("seq_len", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            n_windows=("n_windows", "sum"),
            time_mean=("train_duration_sec", "mean"),
        )
        .fillna(0.0)
        .sort_values("seq_len")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    x = np.arange(len(agg))
    w = 0.28
    ax = axes[0]
    b1 = ax.bar(x - w, agg["macro_f1_mean"], width=w, yerr=agg["macro_f1_std"], capsize=3, color="#1b9e77", label="Macro-F1")
    b2 = ax.bar(x, agg["acc_mean"], width=w, yerr=agg["acc_std"], capsize=3, color="#2b5c8f", label="Accuracy")
    b3 = ax.bar(x + w, agg["mcc_mean"], width=w, color="#d95f02", label="MCC")
    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seq_len={int(r.seq_len)}\n(OOF W={int(r.n_windows)})" for r in agg.itertuples(index=False)])
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel("5-Fold Window Mean Score")
    ax.set_title("Window-Level Evaluation (5-Fold Mean ± SD)")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax2 = axes[1]
    if not abl5_lvl_df.empty:
        lv = abl5_lvl_df.sort_values("seq_len")
        x2 = np.arange(len(lv))
        s1 = ax2.bar(x2 - w, lv["seq_macro_f1"], width=w, color="#1b9e77", label="Seq Macro-F1")
        s2 = ax2.bar(x2, lv["seq_accuracy"], width=w, color="#2b5c8f", label="Seq Accuracy")
        s3 = ax2.bar(x2 + w, lv["seq_mcc"], width=w, color="#d95f02", label="Seq MCC")
        for bars in (s1, s2, s3):
            for rect in bars:
                h = rect.get_height()
                ax2.text(rect.get_x() + rect.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f"seq_len={int(r.seq_len)}\n(N={int(r.n_sequences)} seqs)" for r in lv.itertuples(index=False)])
        ax2.set_ylim(-0.05, 1.12)
        ax2.set_ylabel("Sequence-Level OOF Score")
        ax2.set_title("Sequence-Level OOF Evaluation (Mean Prob Aggregation)")
        ax2.legend(frameon=False, loc="lower left")
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
    else:
        ax2.text(0.5, 0.5, "No sequence-level aggregation rows", ha="center", va="center")
        ax2.axis("off")

    fig.suptitle("Ablation 5: Controlled OCT Sequence Window Length (seq_len = 4 vs 8 vs 12, ResNet50+BiLSTM)", y=1.03)
    return _save_dual(fig, fig_dir, pub_dir, "abl05_oct_sequence_length_comparison")


def _plot_abl6_efficiency_tradeoff(
    abl1: pd.DataFrame,
    abl2: pd.DataFrame,
    abl3: pd.DataFrame,
    fig_dir: Path,
    pub_dir: Path,
) -> list[Path]:
    combined = pd.concat([abl1, abl2, abl3], ignore_index=True).drop_duplicates(subset=["track", "model"])
    if combined.empty or "train_time_mean_sec" not in combined.columns:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "Insufficient runtime/parameter columns for tradeoff plot", ha="center", va="center")
        ax.axis("off")
        return _save_dual(fig, fig_dir, pub_dir, "abl06_ablation_efficiency_tradeoff")

    cat_colors = {"cnn": "#2b5c8f", "hybrid": "#1b9e77", "rnn": "#d95f02", "baseline": "#7570b3"}
    markers = {"fundus_3class": "o", "oct_individual": "s", "oct_sequence": "^", "oct_severity": "D"}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for r in combined.itertuples(index=False):
        params_m = float(getattr(r, "total_params", 0.0)) / 1e6
        t_sec = float(getattr(r, "train_time_mean_sec", 0.0))
        f1 = float(getattr(r, "macro_f1_mean", 0.0))
        cat = str(getattr(r, "category", "cnn"))
        trk = str(getattr(r, "track", ""))
        ax.scatter(
            max(0.1, t_sec),
            f1,
            s=max(45.0, min(280.0, params_m * 7.0 + 40.0)),
            c=cat_colors.get(cat, "#555555"),
            marker=markers.get(trk, "o"),
            alpha=0.82,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(f"{r.model} ({trk.replace('oct_', '').replace('_3class', '')})", (max(0.1, t_sec), f1), fontsize=6.8, alpha=0.85, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Mean Fold Training Time (seconds)")
    ax.set_ylabel("5-Fold Mean Macro-F1")
    ax.set_title("Ablation Overview: Compute Time & Model Scale (Marker Size = Params) vs 5-Fold Macro-F1")
    ax.grid(True, linestyle="--", alpha=0.35)
    return _save_dual(fig, fig_dir, pub_dir, "abl06_ablation_efficiency_tradeoff")


def generate_all_ablation_figures(
    abl1: pd.DataFrame,
    abl2: pd.DataFrame,
    abl3: pd.DataFrame,
    abl4_df: pd.DataFrame,
    abl5_df: pd.DataFrame,
    abl5_lvl_df: pd.DataFrame,
    fig_dir: Path,
    pub_dir: Path,
) -> list[Path]:
    """Generate PNG+PDF figures for all 5 controlled ablations and efficiency overview."""
    _apply_style()
    fig_dir = Path(fig_dir)
    pub_dir = Path(pub_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    paths.extend(
        _plot_summary_grouped_bars(
            abl1,
            ["resnet50", "densenet121", "efficientnet_b0"],
            {"resnet50": "#2b5c8f", "densenet121": "#1b9e77", "efficientnet_b0": "#e6ab02"},
            "Ablation 1: CNN Backbone Comparison",
            "Identical 5-Fold CV Splits per Experimental Track",
            fig_dir,
            pub_dir,
            "abl01_cnn_backbone_comparison",
        )
    )
    paths.extend(
        _plot_summary_grouped_bars(
            abl2,
            ["lstm", "bilstm"],
            {"lstm": "#d95f02", "bilstm": "#7570b3"},
            "Ablation 2: Unidirectional LSTM vs Bidirectional BiLSTM",
            "Identical Input Representations Across 5-Fold CV",
            fig_dir,
            pub_dir,
            "abl02_rnn_lstm_vs_bilstm",
        )
    )
    paths.extend(
        _plot_summary_grouped_bars(
            abl3,
            ["resnet50", "resnet50_lstm", "resnet50_bilstm", "efficientnet_b0", "efficientnet_b0_bilstm"],
            {
                "resnet50": "#2b5c8f",
                "resnet50_lstm": "#66a61e",
                "resnet50_bilstm": "#1b9e77",
                "efficientnet_b0": "#e6ab02",
                "efficientnet_b0_bilstm": "#e7298a",
            },
            "Ablation 3: CNN vs Hybrid CNN+LSTM vs CNN+BiLSTM",
            "Comparable 2D Tracks (Fundus, Individual OCT, Exploratory Severity)",
            fig_dir,
            pub_dir,
            "abl03_hybrid_cnn_rnn_comparison",
        )
    )
    paths.extend(_plot_abl4_trainability(abl4_df, fig_dir, pub_dir))
    paths.extend(_plot_abl5_sequence_length(abl5_df, abl5_lvl_df, fig_dir, pub_dir))
    paths.extend(_plot_abl6_efficiency_tradeoff(abl1, abl2, abl3, fig_dir, pub_dir))
    return paths

