"""CLI runner for Phase 7 controlled ablation studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation.runner import run_ablation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 controlled ablation studies (5-fold CV).")
    parser.add_argument("--eval-dir", default="results/evaluation", help="Directory containing Phase 5 CV evaluation CSVs.")
    parser.add_argument("--output-dir", default="results/ablation_study", help="Directory for ablation CSVs, predictions, and figures.")
    parser.add_argument("--ckpt-dir", default="checkpoints_ablation", help="Directory for new controlled ablation checkpoints.")
    parser.add_argument("--pub-fig-dir", default="results/publication_figures", help="Directory to copy publication PNG+PDF figures.")
    parser.add_argument("--skip-new-training", action="store_true", help="Compile baselines and plots only without training frozen/partial or seq_len 4/12 runs.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run controlled ablation folds even if already completed.")
    args = parser.parse_args()

    manifest = run_ablation_suite(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        pub_fig_dir=args.pub_fig_dir,
        skip_new_training=args.skip_new_training,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

