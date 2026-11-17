"""CLI script for Phase 3 handcrafted and standalone deep feature extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extraction import run_feature_extraction_pipeline
from src.utils import load_all_configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract handcrafted and deep CNN embeddings separately for Fundus and OCT.")
    parser.add_argument("--output-dir", default="results/feature_extraction", help="Destination directory for CSV/NPZ feature files and figures.")
    args = parser.parse_args()

    cfg = load_all_configs()
    manifest = run_feature_extraction_pipeline(cfg=cfg, output_dir=args.output_dir)
    print(f"Fundus handcrafted dim={manifest['fundus_handcrafted_dim']} (HOG={manifest['fundus_hog_dim']})")
    print(f"OCT handcrafted dim={manifest['oct_handcrafted_dim']} (HOG={manifest['oct_hog_dim']})")
    print(f"Extracted {len(manifest['deep_embeddings'])} CNN embedding sets; generated {len(manifest['figures_generated'])} figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

