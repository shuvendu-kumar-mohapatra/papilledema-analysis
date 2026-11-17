"""CLI script to generate CV folds, verify splits, and execute 5-fold CV training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_validation.runner import collect_cv_runs_summary, run_cv_suite
from src.cross_validation.splits import build_and_validate_cv_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 5-fold cross-validation suite.")
    parser.add_argument("--splits-only", action="store_true", help="Only generate and validate 5-fold CV splits without training.")
    parser.add_argument("--tracks", nargs="+", default=None, help="Subset of tracks to run (default: all 4).")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models to run.")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Subset of folds 1..5 to run.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs (e.g. 1 for smoke check).")
    parser.add_argument("--overwrite", action="store_true", help="Re-run completed folds.")
    parser.add_argument("--tag", default="", help="Optional experiment tag (e.g. smoke).")
    args = parser.parse_args()

    out = build_and_validate_cv_splits()
    print("CV split validation passed:", {k: v["validation_passed"] for k, v in out["diagnostics"].items()})
    if args.splits_only:
        return 0

    df = run_cv_suite(
        tracks=args.tracks,
        models=args.models,
        folds=args.folds,
        epochs=args.epochs,
        overwrite=args.overwrite,
        tag=args.tag,
    )
    print(f"CV summary rows completed: {len(df)}")
    if not df.empty:
        print(df.groupby("track").agg(runs=("cv_exp_id", "count"), mean_macro_f1=("test_macro_f1", "mean")).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

