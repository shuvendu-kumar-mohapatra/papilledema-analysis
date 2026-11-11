"""Statistical analysis package exports and full pipeline runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .confidence_intervals import grouped_bootstrap_ci
from .descriptive import compute_descriptive_statistics
from .hypothesis_tests import (
    cohens_d_paired,
    compare_models_for_track,
    holm_bonferroni,
    mcnemar_test,
    rank_biserial_paired,
)

__all__ = [
    "compute_descriptive_statistics",
    "grouped_bootstrap_ci",
    "holm_bonferroni",
    "cohens_d_paired",
    "rank_biserial_paired",
    "mcnemar_test",
    "compare_models_for_track",
    "run_statistical_analysis",
]


def run_statistical_analysis(
    runs_df: pd.DataFrame,
    predictions_dir: str | Path,
    output_dir: str | Path,
    track_classes: dict[str, list[str]],
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run descriptive stats, grouped bootstrap CIs, and hypothesis tests separately per track."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(predictions_dir)

    if runs_df.empty:
        empty = pd.DataFrame()
        for name in ("descriptive_statistics.csv", "confidence_intervals.csv", "hypothesis_tests.csv", "pairwise_comparisons.csv", "effect_sizes.csv"):
            empty.to_csv(out / name, index=False)
        (out / "statistical_summary.md").write_text("# Statistical Summary\n\nNo completed experiments found.\n", encoding="utf-8")
        return {}

    desc_df = compute_descriptive_statistics(runs_df, alpha=alpha)
    desc_df.to_csv(out / "descriptive_statistics.csv", index=False)

    ci_rows: list[dict[str, Any]] = []
    hyp_dfs: list[pd.DataFrame] = []
    pair_dfs: list[pd.DataFrame] = []
    eff_dfs: list[pd.DataFrame] = []

    for track, tdf in runs_df.groupby("track"):
        classes = track_classes.get(str(track), ["Normal", "Papilledema", "Pseudopapilledema"])
        # Select representative prediction CSV per model (highest val macro F1 seed) for McNemar + bootstrap
        best_pred_by_model: dict[str, pd.DataFrame] = {}
        for model_name, mdf in tdf.groupby("model"):
            best_row = mdf.sort_values(["best_val_macro_f1", "seed"], ascending=[False, True]).iloc[0]
            pfile = pred_dir / f"{best_row['exp_id']}_predictions.csv"
            if pfile.exists():
                pdf = pd.read_csv(pfile)
                best_pred_by_model[str(model_name)] = pdf
                ci_list = grouped_bootstrap_ci(
                    pdf, class_names=classes, n_resamples=n_bootstrap, alpha=alpha, seed=seed
                )
                for item in ci_list:
                    ci_rows.append({
                        "track": track,
                        "model": model_name,
                        "exp_id": best_row["exp_id"],
                        **item,
                    })

        res = compare_models_for_track(
            tdf, best_pred_by_model, track_name=str(track), metric="macro_f1", alpha=alpha
        )
        if not res["hypothesis_tests"].empty:
            hyp_dfs.append(res["hypothesis_tests"])
        if not res["pairwise_comparisons"].empty:
            pair_dfs.append(res["pairwise_comparisons"])
        if not res["effect_sizes"].empty:
            eff_dfs.append(res["effect_sizes"])

    ci_df = pd.DataFrame(ci_rows)
    hyp_df = pd.concat(hyp_dfs, ignore_index=True) if hyp_dfs else pd.DataFrame()
    pair_df = pd.concat(pair_dfs, ignore_index=True) if pair_dfs else pd.DataFrame()
    eff_df = pd.concat(eff_dfs, ignore_index=True) if eff_dfs else pd.DataFrame()

    ci_df.to_csv(out / "confidence_intervals.csv", index=False)
    hyp_df.to_csv(out / "hypothesis_tests.csv", index=False)
    pair_df.to_csv(out / "pairwise_comparisons.csv", index=False)
    eff_df.to_csv(out / "effect_sizes.csv", index=False)

    md_lines = [
        "# Statistical Analysis Summary",
        "",
        f"- Significance threshold: alpha = {alpha}",
        f"- Bootstrap resamples (grouped & stratified): {n_bootstrap}",
        "- Multiple comparison adjustment: Holm-Bonferroni within each track/test family",
        "- Caveat (Seeds vs Cohorts): Multi-seed tests (Paired t-test, Wilcoxon, Friedman) capture optimization variance on fixed dataset splits, NOT independent patient cohorts.",
        "- Caveat (Wilcoxon Power): At n=5 seeds, two-sided Wilcoxon signed-rank has minimum achievable p = 0.0625 (> 0.05 before Holm correction).",
        "- Caveat (McNemar Clustering): Sample-level McNemar assumes independent test errors; clustered OCT slices (from few inferred sequences) violate i.i.d. assumptions and are flagged in pairwise notes.",
        "",
    ]
    if not hyp_df.empty:
        sig_count = int(hyp_df["significant"].sum()) if "significant" in hyp_df.columns else 0
        md_lines.append(f"- Total hypothesis tests conducted: {len(hyp_df)} (significant at Holm-adjusted alpha={alpha}: {sig_count})")
    (out / "statistical_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "descriptive_statistics": desc_df,
        "confidence_intervals": ci_df,
        "hypothesis_tests": hyp_df,
        "pairwise_comparisons": pair_df,
        "effect_sizes": eff_df,
    }

