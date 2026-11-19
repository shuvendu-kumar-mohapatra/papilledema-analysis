"""Aggregate all 350 valid 10-fold CV experiment results into publication-ready statistical tables."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results" / "cross_validation" / "10_fold" / "cv_10fold_runs_summary.csv"
OUT_CSV = ROOT / "results" / "cross_validation" / "10_fold" / "cv_10fold_models_aggregated.csv"


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} 10-fold runs from {SUMMARY_CSV}")

    # Metrics to aggregate
    metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr",
        "train_duration_sec",
        "qwk",
        "mae",
    ]

    grouped = df.groupby(["track", "model"])

    rows = []
    for (track, model), group in grouped:
        n_folds = len(group)
        row = {
            "track": track,
            "model": model,
            "n_folds": n_folds,
        }
        for m in metrics:
            if m in group.columns:
                series = group[m].dropna()
                if len(series) > 0:
                    row[f"{m}_mean"] = round(float(series.mean()), 4)
                    row[f"{m}_std"] = round(float(series.std()), 4)
                    row[f"{m}_str"] = f"{series.mean():.4f} +/- {series.std():.4f}"
                else:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan
                    row[f"{m}_str"] = "N/A"
        rows.append(row)

    agg_df = pd.DataFrame(rows)
    # Sort logically by track and model
    agg_df.sort_values(by=["track", "model"], inplace=True)
    agg_df.to_csv(OUT_CSV, index=False)
    print(f"Successfully saved aggregated 10-fold metrics ({len(agg_df)} models) to {OUT_CSV}")

    # Print summary per track
    for trk in agg_df["track"].unique():
        print(f"\n=================================================================")
        print(f" TRACK: {trk.upper()} (10-Fold CV Performance)")
        print(f"=================================================================")
        sub = agg_df[agg_df["track"] == trk]
        cols_to_show = ["model", "n_folds", "accuracy_str", "macro_f1_str", "roc_auc_ovr_str"]
        if trk == "oct_severity":
            cols_to_show.extend(["qwk_str", "mae_str"])
        print(sub[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
