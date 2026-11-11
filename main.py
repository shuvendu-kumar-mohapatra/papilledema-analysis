"""Unified CLI entrypoint for the Papilledema research pipeline."""

from __future__ import annotations

import argparse
import json

from src.data import build_and_save_all_splits, run_full_data_verification
from src.experiments import ExperimentManager
from src.training import run_experiment_suite
from src.utils import get_hardware_info, load_all_configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Papilledema Research Pipeline")
    parser.add_argument("action", choices=["verify-env", "verify-data", "splits", "smoke", "run", "report"])
    parser.add_argument("--track", default="fundus_3class", choices=["fundus_3class", "oct_individual", "oct_sequence", "oct_severity"])
    args = parser.parse_args()

    cfg = load_all_configs()
    if args.action == "verify-env":
        print(json.dumps(get_hardware_info(), indent=2))
    elif args.action == "verify-data":
        print(json.dumps(run_full_data_verification(cfg), indent=2))
    elif args.action == "splits":
        print(json.dumps(build_and_save_all_splits(cfg), indent=2))
    elif args.action == "smoke":
        run_experiment_suite(cfg, track=args.track, models=["efficientnet_b0"], seeds=[42], epochs=1, is_smoke=True, tag="smoke")
        print(ExperimentManager(cfg).generate_all_outputs())
    elif args.action == "run":
        run_experiment_suite(cfg, track=args.track)
        print(ExperimentManager(cfg).generate_all_outputs())
    elif args.action == "report":
        print(ExperimentManager(cfg).generate_all_outputs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

