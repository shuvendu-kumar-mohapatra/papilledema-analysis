"""Run sequential multi-model / multi-seed experiments for a research track."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import ExperimentManager
from src.training import TRACK_TO_TASK, run_experiment_suite
from src.utils import load_all_configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment suite sequentially.")
    parser.add_argument("--dataset", required=True, choices=["fundus", "oct", "all", *TRACK_TO_TASK.keys()])
    parser.add_argument("--task", default="individual", choices=["individual", "sequence", "severity", "all"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_all_configs()
    if args.dataset == "all":
        tracks = ["fundus_3class", "oct_individual", "oct_sequence", "oct_severity"]
    elif args.dataset == "fundus":
        tracks = ["fundus_3class"]
    elif args.dataset == "oct":
        if args.task == "all":
            tracks = ["oct_individual", "oct_sequence", "oct_severity"]
        else:
            tracks = [f"oct_{args.task}"]
    else:
        tracks = [args.dataset]

    seeds = args.seeds if args.seeds else (list(cfg["project"]["random_seeds"]) if args.multi_seed else [int(cfg["project"]["default_seed"])])
    for trk in tracks:
        run_experiment_suite(
            cfg=cfg,
            track=trk,
            models=args.models,
            seeds=seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            is_smoke=args.smoke,
            tag="smoke" if args.smoke else "",
        )
    out = ExperimentManager(cfg).generate_all_outputs()
    print("Generated outputs:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

