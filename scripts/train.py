"""Train and test-evaluate a single model on a chosen experimental track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import SUPPORTED_MODELS
from src.training import TRACK_TO_TASK, run_single_experiment
from src.utils import load_all_configs


def _resolve_track(dataset: str, task: str | None) -> str:
    if dataset in TRACK_TO_TASK:
        return dataset
    if dataset == "fundus":
        return "fundus_3class"
    if dataset == "oct":
        if task in (None, "individual", "oct_individual"):
            return "oct_individual"
        if task in ("sequence", "oct_sequence"):
            return "oct_sequence"
        if task in ("severity", "oct_severity"):
            return "oct_severity"
    raise ValueError(f"Cannot resolve track for dataset={dataset}, task={task}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one model on a research track.")
    parser.add_argument("--dataset", required=True, choices=["fundus", "oct", *TRACK_TO_TASK.keys()])
    parser.add_argument("--task", default=None, choices=["individual", "sequence", "severity", "oct_individual", "oct_sequence", "oct_severity"])
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-mode", choices=["frozen", "partial", "full"], default=None)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg = load_all_configs()
    track = _resolve_track(args.dataset, args.task)
    res = run_single_experiment(
        cfg=cfg,
        track=track,
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        use_clahe=True if args.clahe else None,
        backbone_mode=args.backbone_mode,
        overwrite=args.overwrite,
        resume=not args.no_resume,
        is_smoke=args.smoke,
        tag=args.tag or ("smoke" if args.smoke else ""),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

