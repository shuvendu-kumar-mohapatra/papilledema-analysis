"""CLI script to execute Phase 2 preprocessing analysis and export CSVs & PNG/PDF figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import run_preprocessing_comparison
from src.utils import load_all_configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 before-and-after preprocessing comparison.")
    parser.add_argument("--output-dir", default="results/preprocessing", help="Output directory for preprocessing CSVs and figures.")
    args = parser.parse_args()

    cfg = load_all_configs()
    manifest = run_preprocessing_comparison(cfg=cfg, output_dir=args.output_dir)
    print(f"Analyzed Fundus={manifest['fundus_images_analyzed']}, OCT={manifest['oct_images_analyzed']}, Sequences={manifest['oct_sequence_windows_whole']}")
    print(f"Generated {len(manifest['figures_generated'])} figure files in {manifest['figures_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

