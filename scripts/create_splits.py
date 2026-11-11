"""Create and validate leakage-aware grouped splits for Fundus and OCT tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_and_save_all_splits
from src.utils import load_all_configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Create leakage-aware research splits.")
    parser.add_argument("--dataset", choices=["all", "fundus", "oct"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force-verify", action="store_true")
    args = parser.parse_args()

    cfg = load_all_configs()
    report = build_and_save_all_splits(cfg, seed=args.seed, force_verify=args.force_verify)
    if args.dataset == "fundus":
        print(json.dumps(report["fundus_3class"], indent=2))
    elif args.dataset == "oct":
        print(json.dumps({"oct_3class": report["oct_3class"], "oct_severity": report["oct_severity"]}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

