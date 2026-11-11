"""Generate all CSV tables, statistical summaries, plots, and Markdown reports."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import ExperimentManager
from src.utils import load_all_configs


def main() -> int:
    cfg = load_all_configs()
    out = ExperimentManager(cfg).generate_all_outputs()
    print("Report generation complete:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

