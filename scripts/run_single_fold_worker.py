"""Standalone CLI worker to execute an isolated cross-validation fold with strict resource limits."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.utils.config import load_all_configs, resolve_path
from src.utils.device import get_device, clear_gpu_memory
from scripts.run_experiments_pipeline import run_generic_cv_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single isolated CV fold.")
    parser.add_argument("--track", type=str, required=True, help="Track name (e.g. fundus_3class)")
    parser.add_argument("--model", type=str, required=True, help="Model architecture name")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (1..n_folds)")
    parser.add_argument("--n-folds", type=int, default=10, help="Total folds (10 or 15)")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--num-threads", type=int, default=2, help="CPU threads per worker")
    parser.add_argument("--overwrite", action="store_true", help="Force retrain completed fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Apply strict per-process thread and memory optimization
    torch.set_num_threads(max(1, args.num_threads))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    cfg = load_all_configs()
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)

    out_root = resolve_path(f"results/cross_validation/{args.n_folds}_fold")
    splits_p = out_root / "splits" / f"{args.track}_cv_folds.csv"
    if not splits_p.exists():
        err = f"Splits file not found at {splits_p}"
        print(json.dumps({"status": "failed", "track": args.track, "model": args.model, "fold": args.fold, "error": err}))
        sys.exit(1)

    cv_df = pd.read_csv(splits_p)

    try:
        res = run_generic_cv_fold(
            cfg=cfg,
            cv_df=cv_df,
            track=args.track,
            model_name=args.model,
            fold=args.fold,
            n_folds=args.n_folds,
            output_root=out_root,
            use_augmented=False,
            epochs=args.epochs,
            overwrite=args.overwrite,
            seed=args.seed,
        )
        f1 = float(res.get("macro_f1", 0.0))
        acc = float(res.get("accuracy", 0.0))
        dur = float(res.get("train_duration_sec", 0.0))
        output_payload = {
            "status": "completed",
            "track": args.track,
            "model": args.model,
            "fold": args.fold,
            "n_folds": args.n_folds,
            "macro_f1": f1,
            "accuracy": acc,
            "duration_sec": dur,
        }
        print("__WORKER_RESULT__" + json.dumps(output_payload), flush=True)
        sys.stdout.flush()
        sys.exit(0)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        output_payload = {
            "status": "failed",
            "track": args.track,
            "model": args.model,
            "fold": args.fold,
            "n_folds": args.n_folds,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print("__WORKER_RESULT__" + json.dumps(output_payload), flush=True)
        sys.stderr.flush()
        sys.exit(1)
    finally:
        clear_gpu_memory()


if __name__ == "__main__":
    main()
