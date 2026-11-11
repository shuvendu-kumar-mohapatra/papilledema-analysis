"""Verify Fundus and OCT datasets, detect duplicates/IR-SLO, and write manifests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import run_full_data_verification
from src.utils import load_all_configs


def main() -> int:
    cfg = load_all_configs()
    audit = run_full_data_verification(cfg)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

