"""CLI script for Phase 5 comprehensive 5-fold CV evaluation and figure export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cv_metrics import run_cv_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate completed 5-fold CV runs and generate Phase 5 tables & figures.")
    parser.add_argument("--output-dir", default="results/evaluation", help="Destination directory for evaluation CSVs and figures.")
    parser.add_argument("--pub-fig-dir", default="results/publication_figures", help="Destination directory for publication PNG/PDF copies.")
    args = parser.parse_args()

    manifest = run_cv_evaluation(output_dir=args.output_dir, pub_fig_dir=args.pub_fig_dir)
    print(f"Evaluated {manifest['completed_cv_runs']} CV runs across tracks={manifest['tracks']}")
    print(f"Generated {manifest['figures_generated_count']} figure files in {manifest['output_dir']} and {manifest['publication_figures_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

