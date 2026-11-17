"""CLI script for Phase 6 CV statistical analysis (strictly separate from original results)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics.cv_statistics import run_cv_statistical_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 6 statistical comparisons on 5-fold CV results.")
    parser.add_argument("--eval-dir", default="results/evaluation", help="Phase 5 evaluation directory.")
    parser.add_argument("--output-dir", default="results/cross_validation/statistical_analysis", help="Output directory (never overwrites original results/statistical_analysis/).")
    parser.add_argument("--pub-fig-dir", default="results/publication_figures", help="Publication figures copy directory.")
    args = parser.parse_args()

    manifest = run_cv_statistical_analysis(eval_dir=args.eval_dir, output_dir=args.output_dir, pub_fig_dir=args.pub_fig_dir)
    print(f"CV statistical analysis complete: Friedman={manifest['friedman_rows']}, OOF McNemar sig={manifest['mcnemar_holm_sig_count']}/{manifest['mcnemar_oof_rows']}, N-B Macro-F1 sig={manifest['nadeau_bengio_macro_f1_holm_sig']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

