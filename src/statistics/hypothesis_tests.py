"""Hypothesis testing, Holm-Bonferroni correction, McNemar, Friedman, and effect sizes."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    clean = [float("nan") if v is None else float(v) for v in p_values]
    indexed = sorted(enumerate(clean), key=lambda t: (np.isnan(t[1]), t[1]))
    adjusted = [float("nan")] * m
    running_max = 0.0
    valid_list = [(orig_i, p) for orig_i, p in indexed if not np.isnan(p)]
    mv = len(valid_list)
    for rank, (orig_i, p) in enumerate(valid_list):
        adj = min(1.0, p * (mv - rank))
        running_max = max(running_max, adj)
        adjusted[orig_i] = float(running_max)
    return adjusted


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Compute paired Cohen's d (mean difference / std of differences)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if len(diff) < 2:
        return None
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def rank_biserial_paired(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Compute matched-pairs rank-biserial correlation r = (W+ - W-) / total_rank_sum."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    nz = diff[diff != 0.0]
    n = len(nz)
    if n == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0]))
    w_minus = float(np.sum(ranks[nz < 0]))
    total = n * (n + 1) / 2.0
    return float((w_plus - w_minus) / total)


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, Any]:
    """Run exact or continuity-corrected McNemar's test on paired model correctness."""
    ca = np.asarray(correct_a, dtype=bool)
    cb = np.asarray(correct_b, dtype=bool)
    if len(ca) != len(cb):
        raise ValueError("McNemar inputs must have equal length.")
    # b = A correct & B wrong; c = A wrong & B correct
    b = int(np.sum(ca & (~cb)))
    c = int(np.sum((~ca) & cb))
    n_disc = b + c
    if n_disc == 0:
        return {
            "test_used": "McNemar (exact)",
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_name": "cohens_g",
            "effect_size": 0.0,
            "b_a_only": b,
            "c_b_only": c,
            "notes": "Zero discordant pairs; identical predictions.",
        }
    cohens_g = (b / n_disc) - 0.5
    if n_disc < 25:
        res = stats.binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided")
        return {
            "test_used": "McNemar (exact binomial)",
            "statistic": float(min(b, c)),
            "p_value": float(res.pvalue),
            "effect_size_name": "cohens_g",
            "effect_size": round(float(cohens_g), 6),
            "b_a_only": b,
            "c_b_only": c,
            "notes": f"Discordant count={n_disc} (<25); exact binomial used.",
        }
    chi2 = ((abs(b - c) - 1.0) ** 2) / n_disc
    pval = float(stats.chi2.sf(chi2, df=1))
    return {
        "test_used": "McNemar (chi2 continuity-corrected)",
        "statistic": round(float(chi2), 6),
        "p_value": pval,
        "effect_size_name": "cohens_g",
        "effect_size": round(float(cohens_g), 6),
        "b_a_only": b,
        "c_b_only": c,
        "notes": f"Discordant count={n_disc}. Assumes test sample error independence.",
    }


def compare_models_for_track(
    track_runs_df: pd.DataFrame,
    pred_dfs_by_model: dict[str, pd.DataFrame],
    track_name: str,
    metric: str = "macro_f1",
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Run Friedman, paired t-test, Wilcoxon, Holm-Bonferroni, effect sizes, and McNemar for one track."""
    hypothesis_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []

    models = sorted(track_runs_df["model"].unique().tolist()) if not track_runs_df.empty else []

    # Pivot matched seeds x models
    if not track_runs_df.empty and len(models) >= 2:
        piv = track_runs_df.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        complete_piv = piv.dropna(axis=0, how="any")
        n_seeds = len(complete_piv)

        # 1. Overall Friedman test across >=3 models on >=2 matched seeds
        if len(models) >= 3 and n_seeds >= 2:
            arrays = [complete_piv[m].to_numpy() for m in models]
            try:
                stat, pval = stats.friedmanchisquare(*arrays)
                kendall_w = float(stat / (n_seeds * (len(models) - 1))) if n_seeds > 0 else 0.0
                hypothesis_rows.append({
                    "track": track_name,
                    "models_compared": ", ".join(models),
                    "metric": metric,
                    "test_used": "Friedman test",
                    "statistic": round(float(stat), 6) if not np.isnan(stat) else None,
                    "p_value": float(pval) if not np.isnan(pval) else None,
                    "p_adjusted": float(pval) if not np.isnan(pval) else None,
                    "effect_size_name": "kendalls_w",
                    "effect_size": round(kendall_w, 6) if not np.isnan(kendall_w) else None,
                    "significant": bool(pval < alpha) if not np.isnan(pval) else False,
                    "n_matched_seeds": int(n_seeds),
                    "sample_size": f"{n_seeds} seeds x {len(models)} models",
                    "null_hypothesis": f"All {len(models)} architectures share the same median {metric} across seeds.",
                    "alternative_hypothesis": f"At least one architecture differs in {metric} across matched seeds.",
                    "assumptions": "Matched blocks across random seeds on fixed split; NOT independent patient cohorts.",
                    "interpretation": (
                        f"Reject H0 after global test (W={kendall_w:.3f}, p={pval:.4f}); inspect seed/cohort caveats."
                        if not np.isnan(pval) and pval < alpha
                        else f"Fail to reject H0 (p={pval:.4f}); differences across seeds may reflect stochastic variance."
                    ),
                    "notes": "Matched across random seeds on fixed split (not independent cohorts).",
                })
            except Exception as exc:
                hypothesis_rows.append({
                    "track": track_name,
                    "models_compared": ", ".join(models),
                    "metric": metric,
                    "test_used": "Friedman test",
                    "statistic": None,
                    "p_value": None,
                    "p_adjusted": None,
                    "effect_size_name": "kendalls_w",
                    "effect_size": None,
                    "significant": False,
                    "n_matched_seeds": int(n_seeds),
                    "sample_size": f"{n_seeds} seeds x {len(models)} models",
                    "null_hypothesis": f"All {len(models)} architectures share the same median {metric} across seeds.",
                    "alternative_hypothesis": f"At least one architecture differs in {metric} across matched seeds.",
                    "assumptions": "Matched blocks across random seeds on fixed split; NOT independent patient cohorts.",
                    "interpretation": f"Friedman test not computable: {exc}",
                    "notes": f"Insufficient variation or error: {exc}",
                })
        else:
            hypothesis_rows.append({
                "track": track_name,
                "models_compared": ", ".join(models),
                "metric": metric,
                "test_used": "Friedman test",
                "statistic": None,
                "p_value": None,
                "p_adjusted": None,
                "effect_size_name": "kendalls_w",
                "effect_size": None,
                "significant": False,
                "n_matched_seeds": int(n_seeds),
                "sample_size": f"{n_seeds} seeds x {len(models)} models",
                "null_hypothesis": f"All {len(models)} architectures share the same median {metric} across seeds.",
                "alternative_hypothesis": f"At least one architecture differs in {metric} across matched seeds.",
                "assumptions": "Requires >=3 models and >=2 matched seeds.",
                "interpretation": "Skipped: insufficient matched seeds or models.",
                "notes": f"Insufficient matched seeds (n={n_seeds}) or models (k={len(models)}); >=3 models & >=2 seeds required.",
            })

        # 2. Pairwise seed-based comparisons (paired t-test + Wilcoxon)
        raw_t_pvals = []
        raw_w_pvals = []
        temp_pairs = []
        for ma, mb in combinations(models, 2):
            sub = piv[[ma, mb]].dropna()
            n_pair = len(sub)
            if n_pair < 2:
                temp_pairs.append({
                    "track": track_name, "model_a": ma, "model_b": mb, "metric": metric,
                    "n_pairs": int(n_pair), "mean_diff": None,
                    "t_stat": None, "t_p_value": float("nan"), "t_note": "Need >=2 matched seeds.",
                    "w_stat": None, "w_p_value": float("nan"), "w_note": "Need >=2 matched seeds.",
                    "cohens_d": None, "rank_biserial_r": None,
                })
                raw_t_pvals.append(float("nan"))
                raw_w_pvals.append(float("nan"))
                continue

            a_vals = sub[ma].to_numpy(dtype=np.float64)
            b_vals = sub[mb].to_numpy(dtype=np.float64)
            diff = a_vals - b_vals
            mean_diff = float(np.mean(diff))
            d_eff = cohens_d_paired(a_vals, b_vals)
            rb_eff = rank_biserial_paired(a_vals, b_vals)

            # Normality check if n>=3
            t_note = f"n={n_pair} matched seeds."
            if np.allclose(diff, diff[0]):
                t_stat, t_p = (0.0 if mean_diff == 0 else float("inf")), (1.0 if mean_diff == 0 else 0.0)
                t_note += " Zero variance in paired differences."
            else:
                if n_pair >= 3:
                    _, sw_p = stats.shapiro(diff)
                    if sw_p < 0.05:
                        t_note += f" Shapiro-Wilk p={sw_p:.3f}<0.05 (normality violated)."
                t_stat, t_p = stats.ttest_rel(a_vals, b_vals)

            n_nz = int(np.sum(diff != 0.0))
            min_wilcox_p = 2.0 / (2 ** max(1, n_nz))
            w_note = f"Min achievable two-sided p at n_nonzero={n_nz} (of {n_pair}) is {min_wilcox_p:.4f}."
            if n_nz <= 5:
                w_note += " Insufficient sample size for alpha=0.05."
            if np.all(diff == 0):
                w_stat, w_p = 0.0, 1.0
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        w_stat, w_p = stats.wilcoxon(a_vals, b_vals, zero_method="wilcox")
                except Exception as exc:
                    w_stat, w_p = float("nan"), float("nan")
                    w_note += f" Wilcoxon failed: {exc}"

            raw_t_pvals.append(float(t_p))
            raw_w_pvals.append(float(w_p))
            temp_pairs.append({
                "track": track_name, "model_a": ma, "model_b": mb, "metric": metric,
                "n_pairs": int(n_pair), "mean_diff": round(mean_diff, 6),
                "t_stat": round(float(t_stat), 6) if np.isfinite(t_stat) else None,
                "t_p_value": float(t_p),
                "t_note": t_note,
                "w_stat": round(float(w_stat), 6) if not np.isnan(w_stat) else None,
                "w_p_value": float(w_p),
                "w_note": w_note,
                "cohens_d": round(d_eff, 6) if d_eff is not None else None,
                "rank_biserial_r": round(rb_eff, 6) if rb_eff is not None else None,
            })

        adj_t = holm_bonferroni(raw_t_pvals)
        adj_w = holm_bonferroni(raw_w_pvals)
        for row, pt_adj, pw_adj in zip(temp_pairs, adj_t, adj_w):
            # Add paired t-test and Wilcoxon entries cleanly
            sig_t = bool(not np.isnan(pt_adj) and pt_adj < alpha)
            sig_w = bool(not np.isnan(pw_adj) and pw_adj < alpha)
            n_pairs = int(row["n_pairs"])
            pairwise_rows.append({
                "track": track_name,
                "models_compared": f"{row['model_a']} vs {row['model_b']}",
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "metric": metric,
                "test_used": "Paired t-test (seeds)",
                "sample_size": f"{n_pairs} matched seeds",
                "statistic": row["t_stat"],
                "p_value": None if np.isnan(row["t_p_value"]) else round(row["t_p_value"], 6),
                "p_adjusted": None if np.isnan(pt_adj) else round(pt_adj, 6),
                "effect_size_name": "cohens_d",
                "effect_size": row["cohens_d"],
                "mean_diff": row["mean_diff"],
                "significant": sig_t,
                "null_hypothesis": f"Mean paired seed difference in {metric} ({row['model_a']} - {row['model_b']}) = 0.",
                "alternative_hypothesis": f"Mean paired seed difference in {metric} != 0.",
                "assumptions": "Paired seed differences approximately normal on fixed split (NOT patient cohorts).",
                "interpretation": "Reject H0 (seed-level optimization diff)." if sig_t else "Fail to reject H0 after Holm correction.",
                "notes": row["t_note"],
            })
            pairwise_rows.append({
                "track": track_name,
                "models_compared": f"{row['model_a']} vs {row['model_b']}",
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "metric": metric,
                "test_used": "Wilcoxon signed-rank (seeds)",
                "sample_size": f"{n_pairs} matched seeds",
                "statistic": row["w_stat"],
                "p_value": None if np.isnan(row["w_p_value"]) else round(row["w_p_value"], 6),
                "p_adjusted": None if np.isnan(pw_adj) else round(pw_adj, 6),
                "effect_size_name": "rank_biserial_r",
                "effect_size": row["rank_biserial_r"],
                "mean_diff": row["mean_diff"],
                "significant": sig_w,
                "null_hypothesis": f"Median paired seed difference in {metric} ({row['model_a']} - {row['model_b']}) = 0.",
                "alternative_hypothesis": f"Median paired seed difference in {metric} != 0.",
                "assumptions": "Symmetric paired differences; min achievable two-sided p at n=5 is 0.0625.",
                "interpretation": "Reject H0." if sig_w else "Fail to reject H0 (underpowered at n<=5 seeds).",
                "notes": row["w_note"],
            })
            if row["cohens_d"] is not None:
                effect_rows.append({
                    "track": track_name,
                    "models_compared": f"{row['model_a']} vs {row['model_b']}",
                    "metric": metric,
                    "cohens_d": row["cohens_d"],
                    "rank_biserial_r": row["rank_biserial_r"],
                    "mean_diff": row["mean_diff"],
                })

    # 3. Pairwise McNemar's test on matched test-set predictions
    pred_models = sorted(pred_dfs_by_model.keys())
    mcnemar_raw = []
    mcnemar_pvals = []
    for ma, mb in combinations(pred_models, 2):
        da = pred_dfs_by_model[ma].sort_values("sample_id").reset_index(drop=True)
        db = pred_dfs_by_model[mb].sort_values("sample_id").reset_index(drop=True)
        if len(da) != len(db) or not (da["sample_id"].equals(db["sample_id"])):
            continue
        res = mcnemar_test(da["correct"].to_numpy(), db["correct"].to_numpy())
        n_groups = int(da["group_id"].nunique()) if "group_id" in da.columns else len(da)
        if n_groups < len(da) and (n_groups < 10 or n_groups < 0.7 * len(da)):
            res["notes"] += f" WARNING: {len(da)} samples share {n_groups} clusters/sequences; i.i.d. McNemar is anti-conservative."
        mcnemar_pvals.append(res["p_value"])
        mcnemar_raw.append((ma, mb, res, len(da), n_groups))

    adj_mcn = holm_bonferroni(mcnemar_pvals)
    for (ma, mb, res, n_samples, n_groups), padj in zip(mcnemar_raw, adj_mcn):
        sig = bool(padj < alpha)
        interp = (
            f"Reject H0 (b={res['b_a_only']} vs c={res['c_b_only']} discordant errors, Holm p={padj:.4f})."
            if sig
            else f"Fail to reject H0 (b={res['b_a_only']} vs c={res['c_b_only']}, Holm p={padj:.4f})."
        )
        pairwise_rows.append({
            "track": track_name,
            "models_compared": f"{ma} vs {mb}",
            "model_a": ma,
            "model_b": mb,
            "metric": "test_accuracy_errors",
            "test_used": res["test_used"],
            "sample_size": f"{n_samples} test items ({n_groups} groups)",
            "statistic": res["statistic"],
            "p_value": round(res["p_value"], 6),
            "p_adjusted": round(padj, 6),
            "effect_size_name": res["effect_size_name"],
            "effect_size": res["effect_size"],
            "mean_diff": None,
            "significant": sig,
            "null_hypothesis": f"{ma} and {mb} have equal error proportions on the same test examples.",
            "alternative_hypothesis": f"{ma} and {mb} have different marginal error proportions.",
            "assumptions": "Paired test items with independent errors (best-val-seed checkpoint per model).",
            "interpretation": interp,
            "notes": res["notes"],
        })
        hypothesis_rows.append({
            "track": track_name,
            "models_compared": f"{ma} vs {mb}",
            "metric": "test_accuracy_errors",
            "test_used": res["test_used"],
            "statistic": res["statistic"],
            "p_value": round(res["p_value"], 6),
            "p_adjusted": round(padj, 6),
            "effect_size_name": res["effect_size_name"],
            "effect_size": res["effect_size"],
            "significant": sig,
            "n_matched_seeds": None,
            "sample_size": f"{n_samples} test items ({n_groups} groups)",
            "null_hypothesis": f"{ma} and {mb} have equal error proportions on the same test examples.",
            "alternative_hypothesis": f"{ma} and {mb} have different marginal error proportions.",
            "assumptions": "Paired test items with independent errors (best-val-seed checkpoint per model).",
            "interpretation": interp,
            "notes": res["notes"],
        })

    return {
        "hypothesis_tests": pd.DataFrame(hypothesis_rows),
        "pairwise_comparisons": pd.DataFrame(pairwise_rows),
        "effect_sizes": pd.DataFrame(effect_rows),
    }

