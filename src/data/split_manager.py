"""Leakage-aware grouped stratified dataset splitting and split persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.validation import (
    FUNDUS_CLASSES,
    OCT_3CLASSES,
    OCT_SEVERITY_CLASSES,
    run_full_data_verification,
    validate_no_leakage,
)
from src.utils.config import resolve_path


def _assign_groups_stratified(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    ratios: tuple[float, float, float],
    seed: int,
) -> pd.Series:
    """Assign indivisible groups to train/val/test preserving per-class ratios."""
    rng = np.random.default_rng(seed)
    split_names = ("train", "val", "test")
    group_to_split: dict[str, str] = {}

    for cls, cdf in df.groupby(label_col):
        grp_counts = cdf.groupby(group_col).size().to_dict()
        groups = list(grp_counts.keys())
        rng.shuffle(groups)
        # Largest groups first for tighter ratio matching
        groups.sort(key=lambda g: grp_counts[g], reverse=True)
        total = sum(grp_counts.values())
        targets = {s: r * total for s, r in zip(split_names, ratios)}
        counts = {s: 0 for s in split_names}
        grp_counts_per_split = {s: 0 for s in split_names}

        for g in groups:
            sz = grp_counts[g]
            # Ensure empty splits get filled first when >= 3 groups exist
            empty = [s for s in split_names if grp_counts_per_split[s] == 0]
            if empty and len(groups) >= 3:
                # Prefer val/test for smaller remaining groups, train for largest
                chosen = min(empty, key=lambda s: targets[s] - counts[s]) if sz < total * 0.45 else "train"
                if chosen not in empty:
                    chosen = empty[0]
            else:
                # Choose split with highest remaining target ratio deficit
                chosen = max(split_names, key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1.0))
            group_to_split[g] = chosen
            counts[chosen] += sz
            grp_counts_per_split[chosen] += 1

    return df[group_col].map(group_to_split)


def create_fundus_splits(
    manifest_df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Create grouped stratified train/val/test split keeping near-duplicate clusters intact."""
    df = manifest_df.copy()
    df["split"] = _assign_groups_stratified(df, "cluster_id", "label", ratios, seed)
    df["split_unit_type"] = "near_duplicate_cluster"
    df["split_seed"] = seed
    return df


def _split_single_subject_contiguous(cdf: pd.DataFrame, ratios: tuple[float, float, float]) -> pd.Series:
    """Split a class with <3 independent subject groups via contiguous spatial blocks.

    When 2 series exist, puts one series in train and partitions the 2nd series into
    [train_head | val_block | test_block] so validation separates train from test slices.
    """
    cdf_sorted = cdf.sort_values(["inferred_sequence_id", "slice_index", "image_path"]).copy()
    seqs = cdf_sorted["inferred_sequence_id"].unique().tolist()
    total = len(cdf_sorted)
    n_test = max(1, int(round(total * ratios[2])))
    n_val = max(1, int(round(total * ratios[1])))

    assign = pd.Series("train", index=cdf_sorted.index)
    if len(seqs) >= 2:
        # Keep largest sequence completely in train; block-split the smaller/other sequence
        seq_sizes = cdf_sorted.groupby("inferred_sequence_id").size().sort_values(ascending=False)
        block_seq = seq_sizes.index[-1]
        sub = cdf_sorted[cdf_sorted["inferred_sequence_id"] == block_seq]
        if len(sub) >= n_val + n_test + 2:
            # Take val then test from the tail of block_seq
            val_idx = sub.index[-(n_val + n_test) : -n_test]
            test_idx = sub.index[-n_test:]
            assign.loc[val_idx] = "val"
            assign.loc[test_idx] = "test"
            return assign.reindex(cdf.index)

    # Single-sequence fallback: contiguous [train | val | test] blocks
    n_train = max(1, total - n_val - n_test)
    assign.iloc[n_train : n_train + n_val] = "val"
    assign.iloc[n_train + n_val :] = "test"
    return assign.reindex(cdf.index)


def create_oct_splits(
    manifest_df: pd.DataFrame,
    task: str = "oct_3class",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Create OCT splits for oct_3class or exploratory oct_severity."""
    if task == "oct_3class":
        df = manifest_df.copy()
        df["label"] = df["label_3class"]
        df["label_idx"] = df["label_3class_idx"]
    elif task == "oct_severity":
        mask = manifest_df["label_severity"].notna() & (manifest_df["label_severity"].astype(str).str.strip() != "")
        df = manifest_df[mask].copy().reset_index(drop=True)
        df["label"] = df["label_severity"].astype(str)
        df["label_idx"] = df["label_severity_idx"].astype(int)
    else:
        raise ValueError(f"Unsupported OCT task: {task}")

    df["split"] = ""
    df["split_unit_type"] = ""
    for cls, cdf in df.groupby("label"):
        n_prefixes = cdf["inferred_subject_prefix"].nunique()
        if n_prefixes >= 3:
            assigned = _assign_groups_stratified(cdf, "inferred_subject_prefix", "label", ratios, seed)
            df.loc[cdf.index, "split"] = assigned
            df.loc[cdf.index, "split_unit_type"] = "inferred_subject_prefix_group"
        else:
            assigned = _split_single_subject_contiguous(cdf, ratios)
            df.loc[cdf.index, "split"] = assigned
            df.loc[cdf.index, "split_unit_type"] = "contiguous_slice_block_single_subject"

    df["split_seed"] = seed
    return df


def build_and_save_all_splits(
    cfg: dict[str, Any],
    seed: int | None = None,
    force_verify: bool = False,
) -> dict[str, Any]:
    """Generate and validate Fundus, OCT 3-class, and OCT severity split CSVs."""
    seed = int(seed if seed is not None else cfg["project"]["default_seed"])
    manifests_dir = resolve_path(cfg["paths"]["manifests_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    f_manifest_path = manifests_dir / "fundus_manifest.csv"
    o_manifest_path = manifests_dir / "oct_manifest.csv"
    if force_verify or not f_manifest_path.exists() or not o_manifest_path.exists():
        run_full_data_verification(cfg)

    f_manifest = pd.read_csv(f_manifest_path, keep_default_na=False)
    o_manifest = pd.read_csv(o_manifest_path, keep_default_na=False)

    f_ratios = tuple(cfg["fundus"].get("split_ratios", [0.70, 0.15, 0.15]))
    o_ratios = tuple(cfg["oct"].get("split_ratios", [0.70, 0.15, 0.15]))

    f_splits = create_fundus_splits(f_manifest, ratios=f_ratios, seed=seed)
    f_report = validate_no_leakage(f_splits, "fundus_3class", label_col="label", expected_classes=FUNDUS_CLASSES)
    f_splits.to_csv(splits_dir / "fundus_splits.csv", index=False)

    o3_splits = create_oct_splits(o_manifest, task="oct_3class", ratios=o_ratios, seed=seed)
    o3_report = validate_no_leakage(o3_splits, "oct_3class", label_col="label", expected_classes=OCT_3CLASSES)
    o3_splits.to_csv(splits_dir / "oct_3class_splits.csv", index=False)

    osev_splits = create_oct_splits(o_manifest, task="oct_severity", ratios=o_ratios, seed=seed)
    osev_report = validate_no_leakage(osev_splits, "oct_severity", label_col="label", expected_classes=OCT_SEVERITY_CLASSES)
    osev_splits.to_csv(splits_dir / "oct_severity_splits.csv", index=False)

    validation = {
        "seed": seed,
        "fundus_3class": f_report,
        "oct_3class": o3_report,
        "oct_severity": osev_report,
    }
    with (splits_dir / "split_validation.json").open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    return validation


def load_split_df(cfg: dict[str, Any], task: str) -> pd.DataFrame:
    """Load validated split DataFrame for a given task, creating splits if absent."""
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    mapping = {
        "fundus_3class": splits_dir / "fundus_splits.csv",
        "oct_3class": splits_dir / "oct_3class_splits.csv",
        "oct_severity": splits_dir / "oct_severity_splits.csv",
    }
    if task not in mapping:
        raise ValueError(f"Unknown task '{task}'. Valid: {list(mapping)}")
    p = mapping[task]
    if not p.exists():
        build_and_save_all_splits(cfg)
    df = pd.read_csv(p, keep_default_na=False)
    expected = {
        "fundus_3class": FUNDUS_CLASSES,
        "oct_3class": OCT_3CLASSES,
        "oct_severity": OCT_SEVERITY_CLASSES,
    }[task]
    validate_no_leakage(df, task, label_col="label", expected_classes=expected)
    return df
