"""Phase 6 statistical analysis for 5-fold CV (Nadeau-Bengio, Friedman, OOF McNemar, Holm)."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from src.cross_validation.runner import CV_TRACK_SPECS, collect_cv_runs_summary
from src.statistics.hypothesis_tests import (
    cohens_d_paired,
    holm_bonferroni,
    mcnemar_test,
    rank_biserial_paired,
)
from src.utils.config import load_config, resolve_path

TARGET_METRICS = ["macro_f1", "balanced_accuracy", "macro_recall", "macro_precision", "weighted_f1"]


def nadeau_bengio_corrected_ttest(
    diffs: np.ndarray,
    train_test_ratio: float = 0.25,
) -> tuple[float, float]:
    """Variance-corrected paired t-test accounting for overlapping CV training sets.

    Nadeau & Bengio (2003): Var_corr = (1/k + n_test / n_train_inner) * s^2_d.
    Here outer test fold is ~1/5 and train_inner is ~3/5 -> ratio ~ 1/3 (conservative 0.25-0.333).
    """
    k = len(diffs)
    if k < 2:
        return float("nan"), float("nan")
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    if var_d == 0.0:
        return (0.0 if mean_d == 0.0 else float("inf")), (1.0 if mean_d == 0.0 else 0.0)
    se_corr = np.sqrt((1.0 / k + train_test_ratio) * var_d)
    t_stat = mean_d / se_corr
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=k - 1))
    return float(t_stat), p_val


def _save_fig_pair(fig: plt.Figure, out_dir: Path, pub_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"{name}.png"
    p2 = out_dir / f"{name}.pdf"
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    (pub_dir / p1.name).write_bytes(p1.read_bytes())
    (pub_dir / p2.name).write_bytes(p2.read_bytes())
    return [p1, p2]


def run_cv_statistical_analysis(
    eval_dir: str | Path = "results/evaluation",
    cv_cfg: dict[str, Any] | None = None,
    output_dir: str | Path = "results/cross_validation/statistical_analysis",
    pub_fig_dir: str | Path = "results/publication_figures",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run independent-track CV hypothesis tests without modifying original results/statistical_analysis/."""
    ev_root = resolve_path(eval_dir)
    out_root = resolve_path(output_dir)
    fig_dir = out_root / "figures"
    pub_dir = resolve_path(pub_fig_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cv_raw = cv_cfg or load_config("configs/cross_validation.yaml")
    cv_p = cv_raw.get("cross_validation", cv_raw)
    pred_dir = resolve_path(cv_p["paths"]["predictions_dir"])

    fold_df = pd.read_csv(ev_root / "cv_fold_metrics.csv")
    runs_df = collect_cv_runs_summary(cv_raw)

    friedman_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    mcnemar_rows: list[dict[str, Any]] = []

    for track in sorted(fold_df["track"].unique()):
        t_df = fold_df[fold_df["track"] == track].copy()
        models = sorted(t_df["model"].unique().tolist())

        # 1. Friedman + Kendall's W + Nadeau-Bengio & Wilcoxon across 5 metrics
        for metric in TARGET_METRICS:
            piv = t_df.pivot(index="fold", columns="model", values=metric)[models].dropna()
            k_models = len(models)
            n_folds = len(piv)
            if k_models >= 3 and n_folds >= 2:
                stat_f, p_f = stats.friedmanchisquare(*[piv[m].to_numpy() for m in models])
                w_kendall = float(stat_f / (n_folds * (k_models - 1)))
            else:
                stat_f, p_f, w_kendall = float("nan"), float("nan"), float("nan")
            friedman_rows.append({
                "track": track,
                "metric": metric,
                "n_models": k_models,
                "n_folds": n_folds,
                "friedman_chi2": None if np.isnan(stat_f) else round(float(stat_f), 6),
                "p_value": None if np.isnan(p_f) else float(p_f),
                "kendalls_w": None if np.isnan(w_kendall) else round(w_kendall, 6),
                "significant_unadj": bool(not np.isnan(p_f) and p_f < alpha),
                "caveat": "5 CV folds share overlapping ~75% training subsets; treat as fold-level block omnibus.",
            })

            raw_nb_p, raw_w_p, temp = [], [], []
            for ma, mb in combinations(models, 2):
                a = piv[ma].to_numpy(dtype=np.float64)
                b = piv[mb].to_numpy(dtype=np.float64)
                diff = a - b
                t_nb, p_nb = nadeau_bengio_corrected_ttest(diff, train_test_ratio=1.0 / 3.0)
                if np.all(diff == 0):
                    w_stat, p_w = 0.0, 1.0
                else:
                    try:
                        w_stat, p_w = stats.wilcoxon(a, b, zero_method="wilcox")
                    except Exception:
                        w_stat, p_w = float("nan"), float("nan")
                raw_nb_p.append(p_nb)
                raw_w_p.append(p_w)
                temp.append({
                    "track": track,
                    "metric": metric,
                    "model_a": ma,
                    "model_b": mb,
                    "mean_diff": round(float(np.mean(diff)), 6),
                    "cohens_d": cohens_d_paired(a, b),
                    "rank_biserial_r": rank_biserial_paired(a, b),
                    "nb_t_stat": None if not np.isfinite(t_nb) else round(t_nb, 6),
                    "nb_p_raw": None if np.isnan(p_nb) else float(p_nb),
                    "wilcoxon_stat": None if np.isnan(w_stat) else round(float(w_stat), 6),
                    "wilcoxon_p_raw": None if np.isnan(p_w) else float(p_w),
                })
            adj_nb = holm_bonferroni(raw_nb_p)
            adj_w = holm_bonferroni(raw_w_p)
            for row, p1, p2 in zip(temp, adj_nb, adj_w):
                row["nb_p_holm"] = None if np.isnan(p1) else round(float(p1), 6)
                row["nb_significant_holm"] = bool(not np.isnan(p1) and p1 < alpha)
                row["wilcoxon_p_holm"] = None if np.isnan(p2) else round(float(p2), 6)
                row["wilcoxon_significant_holm"] = bool(not np.isnan(p2) and p2 < alpha)
                row["wilcoxon_note"] = "n=5 folds yields min two-sided p=0.0625 (>0.05)."
                pairwise_rows.append(row)

        # 2. Pooled Out-of-Fold McNemar Tests across full dataset items
        oof_correct: dict[str, pd.Series] = {}
        r_trk = runs_df[runs_df["track"] == track]
        for m_name, grp in r_trk.groupby("model"):
            dfs = []
            for r in grp.sort_values("fold").itertuples(index=False):
                pdf = pd.read_csv(pred_dir / f"{r.cv_exp_id}_predictions.csv")
                dfs.append(pdf[["sample_id", "correct"]])
            cat = pd.concat(dfs, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
            oof_correct[str(m_name)] = cat.set_index("sample_id")["correct"].astype(bool)

        raw_mc_p, temp_mc = [], []
        for ma, mb in combinations(sorted(oof_correct.keys()), 2):
            common_idx = oof_correct[ma].index.intersection(oof_correct[mb].index)
            mc = mcnemar_test(oof_correct[ma].loc[common_idx].to_numpy(), oof_correct[mb].loc[common_idx].to_numpy())
            raw_mc_p.append(mc["p_value"])
            temp_mc.append({
                "track": track,
                "model_a": ma,
                "model_b": mb,
                "n_oof_items": int(len(common_idx)),
                "test_used": mc["test_used"],
                "b_a_only_correct": mc["b_a_only"],
                "c_b_only_correct": mc["c_b_only"],
                "statistic": mc["statistic"],
                "p_value": mc["p_value"],
                "cohens_g": mc["effect_size"],
                "notes": mc["notes"] + (" Non-independent OCT slices/prefix caveats apply." if "oct" in track else ""),
            })
        adj_mc = holm_bonferroni(raw_mc_p)
        for row, p_adj in zip(temp_mc, adj_mc):
            row["p_adjusted_holm"] = round(float(p_adj), 6)
            row["significant_holm"] = bool(p_adj < alpha)
            mcnemar_rows.append(row)

    fried_df = pd.DataFrame(friedman_rows)
    # Holm adjust Friedman across tracks per metric
    fried_df["p_adjusted_holm"] = np.nan
    for metric, idx in fried_df.groupby("metric").groups.items():
        pvals = fried_df.loc[idx, "p_value"].fillna(1.0).tolist()
        fried_df.loc[idx, "p_adjusted_holm"] = holm_bonferroni(pvals)
    fried_df["significant_holm"] = fried_df["p_adjusted_holm"] < alpha

    pair_df = pd.DataFrame(pairwise_rows)
    mc_df = pd.DataFrame(mcnemar_rows)

    fried_df.to_csv(out_root / "cv_friedman_tests.csv", index=False)
    pair_df.to_csv(out_root / "cv_pairwise_comparisons.csv", index=False)
    mc_df.to_csv(out_root / "cv_oof_mcnemar_tests.csv", index=False)

    # 3. Original 5-Seed Fixed-Split vs New 5-Fold CV Comparison Table (Read-Only read of results/tables/)
    orig_files = {
        "fundus_3class": "results/tables/model_comparison.csv",
        "oct_individual": "results/tables/oct_individual_comparison.csv",
        "oct_sequence": "results/tables/oct_sequence_comparison.csv",
        "oct_severity": "results/tables/oct_severity_comparison.csv",
    }
    orig_frames = []
    for trk, rel_p in orig_files.items():
        fp = resolve_path(rel_p)
        if fp.exists():
            odf = pd.read_csv(fp)
            odf["track"] = trk
            orig_frames.append(odf)
    cv_sum = pd.read_csv(ev_root / "cv_model_summary.csv")
    comp_df = pd.DataFrame()
    if orig_frames:
        orig_all = pd.concat(orig_frames, ignore_index=True)
        # Inspect exact columns dynamically so merge works reliably
        m_col = "model" if "model" in orig_all.columns else orig_all.columns[0]
        f1_col = "macro_f1_mean" if "macro_f1_mean" in orig_all.columns else [c for c in orig_all.columns if "macro_f1" in c and "std" not in c][0]
        f1_sd_col = "macro_f1_std" if "macro_f1_std" in orig_all.columns else next((c for c in orig_all.columns if "macro_f1" in c and "std" in c), None)
        sel = orig_all[["track", m_col, f1_col] + ([f1_sd_col] if f1_sd_col else [])].rename(columns={
            m_col: "model", f1_col: "orig_5seed_macro_f1_mean", **({f1_sd_col: "orig_5seed_macro_f1_std"} if f1_sd_col else {})
        })
        comp_df = cv_sum[["track", "model", "category", "macro_f1_mean", "macro_f1_std", "oof_macro_f1"]].merge(sel, on=["track", "model"], how="left")
        comp_df["cv_minus_orig_macro_f1"] = (comp_df["macro_f1_mean"] - comp_df["orig_5seed_macro_f1_mean"]).round(4)
        comp_df.to_csv(out_root / "original_vs_cv_comparison.csv", index=False)

    # 4. Generate Statistical Figures (PNG + PDF)
    figs: list[Path] = []
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: Friedman Kendall's W & -log10(p)
    f_m = fried_df[fried_df["metric"] == "macro_f1"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].bar(f_m["track"], f_m["kendalls_w"], color="#2b5c8f", edgecolor="#222")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Kendall's W Concordance")
    axes[0].set_title("5-Fold Friedman Effect Size (Macro-F1)")
    axes[0].tick_params(axis="x", rotation=18)
    for i, wv in enumerate(f_m["kendalls_w"]):
        axes[0].text(i, wv + 0.02, f"{wv:.3f}", ha="center", fontweight="bold", fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    neglogp = -np.log10(np.clip(f_m["p_adjusted_holm"].to_numpy(dtype=float), 1e-12, 1.0))
    axes[1].bar(f_m["track"], neglogp, color="#1b9e77", edgecolor="#222")
    axes[1].axhline(-np.log10(alpha), color="#cc0000", linestyle="--", label=f"alpha={alpha}")
    axes[1].set_ylabel("-log10(Holm-Adjusted Friedman p)")
    axes[1].set_title("Global Architecture Significance Across Folds")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    fig.suptitle("Phase 6 Omnibus Friedman & Kendall's W Results Across All Tracks (5-Fold CV)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figs.extend(_save_fig_pair(fig, fig_dir, pub_dir, "cv_stats_01_friedman_kendall"))

    # Fig 2: Pairwise Macro-F1 Difference & OOF McNemar Significance Heatmaps per track (2x2 grid)
    tracks_ord = [t for t in ("fundus_3class", "oct_individual", "oct_sequence", "oct_severity") if t in set(mc_df["track"])]
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 13.5))
    for ax, trk in zip(axes.flat, tracks_ord):
        sub_p = pair_df[(pair_df["track"] == trk) & (pair_df["metric"] == "macro_f1")]
        sub_m = mc_df[mc_df["track"] == trk]
        m_list = sorted(set(sub_p["model_a"]).union(set(sub_p["model_b"])))
        mat = pd.DataFrame(0.0, index=m_list, columns=m_list)
        annot = pd.DataFrame("", index=m_list, columns=m_list)
        mc_lookup = {(r.model_a, r.model_b): r for r in sub_m.itertuples(index=False)}
        for r in sub_p.itertuples(index=False):
            d = r.mean_diff
            mat.loc[r.model_a, r.model_b] = d
            mat.loc[r.model_b, r.model_a] = -d
            mcr = mc_lookup.get((r.model_a, r.model_b))
            sig_mc = "*" if (mcr and mcr.significant_holm) else ""
            sig_nb = "†" if r.nb_significant_holm else ""
            annot.loc[r.model_a, r.model_b] = f"{d:+.2f}{sig_mc}{sig_nb}"
            annot.loc[r.model_b, r.model_a] = f"{-d:+.2f}{sig_mc}{sig_nb}"
        sns.heatmap(mat, annot=annot, fmt="", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.75}, annot_kws={"size": 7})
        ax.set_title(f"{trk}: Macro-F1 Diff (* OOF McNemar p_holm<0.05, † N-B t p_holm<0.05)")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle("Pairwise Model Differences (Row - Col Macro-F1) & Holm-Corrected Significance")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figs.extend(_save_fig_pair(fig, fig_dir, pub_dir, "cv_stats_02_pairwise_heatmaps"))

    # Fig 3: Original 5-Seed Fixed-Split vs New 5-Fold CV Comparison
    if not comp_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.5))
        for ax, trk in zip(axes.flat, tracks_ord):
            sub = comp_df[comp_df["track"] == trk].sort_values("macro_f1_mean", ascending=False)
            x = np.arange(len(sub))
            w = 0.36
            ax.bar(x - w / 2, sub["orig_5seed_macro_f1_mean"], w, label="Original Fixed-Split (5 Seeds)", color="#4e79a7", edgecolor="#222")
            ax.bar(x + w / 2, sub["macro_f1_mean"], w, yerr=sub["macro_f1_std"], capsize=2, label="New 5-Fold CV (Mean±SD)", color="#59a14f", edgecolor="#222")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["model"], rotation=32, ha="right", fontsize=8)
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("Macro-F1")
            ax.set_title(f"{trk}: Original 5-Seed vs. 5-Fold CV")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.25, linestyle="--")
        fig.suptitle("Original Fixed-Split (5-Seed) vs. New 5-Fold Cross-Validation Macro-F1")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figs.extend(_save_fig_pair(fig, fig_dir, pub_dir, "cv_stats_03_original_vs_cv_comparison"))

    manifest = {
        "friedman_rows": int(len(fried_df)),
        "pairwise_rows": int(len(pair_df)),
        "mcnemar_oof_rows": int(len(mc_df)),
        "mcnemar_holm_sig_count": int(mc_df["significant_holm"].sum()) if not mc_df.empty else 0,
        "nadeau_bengio_macro_f1_holm_sig": int(pair_df.loc[pair_df["metric"] == "macro_f1", "nb_significant_holm"].sum()) if not pair_df.empty else 0,
        "output_dir": str(out_root),
        "figures_generated": [str(p) for p in figs],
    }
    (out_root / "cv_statistical_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

