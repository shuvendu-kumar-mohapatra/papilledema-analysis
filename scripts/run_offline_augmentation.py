"""CLI runner to execute the offline data augmentation pipeline for Fundus, OCT, and Severity."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.offline_augmentation import generate_offline_augmented_dataset
from src.utils.config import load_all_configs, resolve_path


def main() -> int:
    cfg = load_all_configs()
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    out_root = resolve_path("results/augmentation")
    out_manifests = out_root / "manifests"
    out_summary = out_root / "augmentation_summary"
    out_manifests.mkdir(parents=True, exist_ok=True)
    out_summary.mkdir(parents=True, exist_ok=True)

    print("=== Running Offline Data Augmentation Pipeline ===")

    # 1. Fundus
    print("\n[1/3] Augmenting Fundus 3-Class dataset...")
    fundus_raw = pd.read_csv(splits_dir / "fundus_splits.csv")
    fundus_targets = {"Normal": 778, "Papilledema": 750, "Pseudopapilledema": 750}
    f_man, f_sum = generate_offline_augmented_dataset(
        fundus_raw,
        track="fundus_3class",
        output_root=out_root,
        target_counts=fundus_targets,
        image_size=int(cfg["fundus"]["image_size"]),
        seed=42,
    )
    print(f_sum.to_string(index=False))

    # 2. Individual OCT
    print("\n[2/3] Augmenting Individual OCT dataset...")
    oct_raw = pd.read_csv(splits_dir / "oct_3class_splits.csv")
    oct_targets = {"Normal": 850, "Papilledema": 895, "Pseudopapilledema": 850}
    o_man, o_sum = generate_offline_augmented_dataset(
        oct_raw,
        track="oct_individual",
        output_root=out_root,
        target_counts=oct_targets,
        image_size=int(cfg["oct"]["image_size"]),
        seed=42,
    )
    print(o_sum.to_string(index=False))

    # 3. OCT Severity
    print("\n[3/3] Augmenting OCT Severity dataset...")
    sev_raw = pd.read_csv(splits_dir / "oct_severity_splits.csv")
    # Defensible balanced expansion (G1 from 40 to 400, G2 at 598, G3 from 184 to 550, G4 from 73 to 450)
    sev_targets = {"G1": 400, "G2": 598, "G3": 550, "G4": 450}
    s_man, s_sum = generate_offline_augmented_dataset(
        sev_raw,
        track="oct_severity",
        output_root=out_root,
        target_counts=sev_targets,
        image_size=int(cfg["oct"]["image_size"]),
        seed=42,
    )
    print(s_sum.to_string(index=False))

    # Combine manifests and summaries
    combined_man = pd.concat([f_man, o_man, s_man], ignore_index=True)
    combined_sum = pd.concat([f_sum, o_sum, s_sum], ignore_index=True)

    man_csv = out_manifests / "augmentation_manifest.csv"
    sum_csv = out_summary / "class_expansion_summary.csv"
    combined_man.to_csv(man_csv, index=False)
    combined_sum.to_csv(sum_csv, index=False)

    print(f"\nManifest saved to: {man_csv} ({len(combined_man)} total image entries)")
    print(f"Summary saved to: {sum_csv}")
    print("\nTotal synthetic images generated:")
    print(combined_sum.groupby("dataset")[["original_count", "synthetic_generated", "final_total"]].sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
