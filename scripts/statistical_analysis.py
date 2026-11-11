"""Run statistical analysis across completed experiments for specified or all tracks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.experiments import ExperimentManager
from src.statistics import run_statistical_analysis
from src.utils import load_all_configs, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run statistical analysis on completed runs.")
    parser.add_argument("--dataset", choices=["all", "fundus", "oct"], default="all")
    args = parser.parse_args()

    cfg = load_all_configs()
    mgr = ExperimentManager(cfg)
    runs_df = mgr.collect_completed_runs()
    if args.dataset == "fundus":
        runs_df = runs_df[runs_df["dataset"] == "fundus"]
    elif args.dataset == "oct":
        runs_df = runs_df[runs_df["dataset"] == "oct"]

    res_root = resolve_path(cfg["paths"]["results_dir"])
    out = run_statistical_analysis(
        runs_df,
        predictions_dir=res_root / "predictions",
        output_dir=res_root / "statistical_analysis",
        track_classes={
            "fundus_3class": FUNDUS_CLASSES,
            "oct_individual": OCT_3CLASSES,
            "oct_sequence": OCT_3CLASSES,
            "oct_severity": OCT_SEVERITY_CLASSES,
        },
        alpha=float(cfg["evaluation"]["alpha"]),
        n_bootstrap=int(cfg["evaluation"]["bootstrap_resamples"]),
        seed=int(cfg["project"]["default_seed"]),
    )
    print("Statistical tables generated:", list(out.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

