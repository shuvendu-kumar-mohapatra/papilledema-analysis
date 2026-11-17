"""Methodologically defensible 5-fold CV split generation and strict pre-training validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.datasets import OCTSequenceDataset
from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.utils.config import load_all_configs, load_config, resolve_path


def _assign_prefix_balanced_folds(df: pd.DataFrame, group_col: str = "inferred_subject_prefix", n_folds: int = 5) -> np.ndarray:
    """Assign >=5 indivisible groups cleanly across 5 folds balancing group sizes."""
    grp_sizes = df.groupby(group_col).size().sort_values(ascending=False)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    grp_map: dict[str, int] = {}
    for g, sz in grp_sizes.items():
        f_idx = int(np.argmin(fold_sizes))
        grp_map[str(g)] = f_idx
        fold_sizes[f_idx] += int(sz)
    return df[group_col].map(grp_map).to_numpy(dtype=np.int64)


def _assign_contiguous_block_folds(df: pd.DataFrame, sort_cols: list[str], n_folds: int = 5) -> np.ndarray:
    """Split an ordered table into n_folds contiguous non-interleaved spatial/window blocks."""
    ordered = df.sort_values(sort_cols).reset_index()
    chunks = np.array_split(ordered["index"].to_numpy(), n_folds)
    out = pd.Series(-1, index=df.index, dtype=np.int64)
    for f_idx, idx_arr in enumerate(chunks):
        out.loc[idx_arr] = f_idx
    return out.to_numpy(dtype=np.int64)


def generate_fundus_cv_splits(fundus_df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    df = fundus_df.copy().reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=np.int64)
    for f_idx, (_, test_idx) in enumerate(sgkf.split(df, df["label"], df["cluster_id"])):
        folds[test_idx] = f_idx
    df["cv_fold"] = folds
    df["cv_strategy"] = "StratifiedGroupKFold(cluster_id)"
    df["patient_independent"] = False
    return df


def generate_oct_individual_cv_splits(oct_3_df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    df = oct_3_df.copy().reset_index(drop=True)
    folds = np.full(len(df), -1, dtype=np.int64)
    strategies: list[str] = [""] * len(df)
    for cls, cdf in df.groupby("label"):
        n_pref = cdf["inferred_subject_prefix"].nunique()
        if n_pref >= n_folds:
            f_arr = _assign_prefix_balanced_folds(cdf, "inferred_subject_prefix", n_folds=n_folds)
            folds[cdf.index.to_numpy()] = f_arr
            strat = "inferred_subject_prefix_grouped"
        else:
            f_arr = _assign_contiguous_block_folds(cdf, ["inferred_sequence_id", "slice_index", "image_path"], n_folds=n_folds)
            folds[cdf.index.to_numpy()] = f_arr
            strat = "contiguous_slice_block_single_prefix"
        for i in cdf.index:
            strategies[i] = strat
    df["cv_fold"] = folds
    df["cv_strategy"] = strategies
    df["patient_independent"] = False
    return df


def generate_oct_sequence_cv_splits(
    oct_3_df: pd.DataFrame,
    seq_len: int = 8,
    stride: int = 8,
    min_seq_len: int = 2,
    n_folds: int = 5,
) -> pd.DataFrame:
    ds_all = OCTSequenceDataset(oct_3_df, seq_len=seq_len, stride=stride, min_seq_len=min_seq_len, is_train=False)
    seq_to_prefix = (
        oct_3_df[["inferred_sequence_id", "inferred_subject_prefix"]]
        .drop_duplicates()
        .set_index("inferred_sequence_id")["inferred_subject_prefix"]
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for item in ds_all.sequences:
        lbl = item["label"]
        rows.append({
            "sample_id": item["sample_id"],
            "sequence_id": item["sequence_id"],
            "inferred_subject_prefix": str(seq_to_prefix.get(item["sequence_id"], item["sequence_id"])),
            "label": lbl,
            "label_idx": OCT_3CLASSES.index(lbl),
            "actual_len": len(item["paths"]),
            "start_slice_idx": int(item["slice_indices"][0]),
            "paths_json": json.dumps(item["paths"]),
            "slice_indices_json": json.dumps([int(s) for s in item["slice_indices"]]),
        })
    df = pd.DataFrame(rows)
    folds = np.full(len(df), -1, dtype=np.int64)
    strategies = [""] * len(df)
    for cls, cdf in df.groupby("label"):
        n_pref = cdf["inferred_subject_prefix"].nunique()
        if n_pref >= n_folds:
            f_arr = _assign_prefix_balanced_folds(cdf, "inferred_subject_prefix", n_folds=n_folds)
            folds[cdf.index.to_numpy()] = f_arr
            strat = "inferred_subject_prefix_grouped_windows"
        else:
            f_arr = _assign_contiguous_block_folds(cdf, ["sequence_id", "start_slice_idx", "sample_id"], n_folds=n_folds)
            folds[cdf.index.to_numpy()] = f_arr
            strat = "contiguous_window_block_2seqs"
        for i in cdf.index:
            strategies[i] = strat
    df["cv_fold"] = folds
    df["cv_strategy"] = strategies
    df["patient_independent"] = False
    return df


def generate_oct_severity_cv_splits(oct_sev_df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    df = oct_sev_df.copy().reset_index(drop=True)
    folds = np.full(len(df), -1, dtype=np.int64)
    strategies = [""] * len(df)
    for cls, cdf in df.groupby("label"):
        n_pref = cdf["inferred_subject_prefix"].nunique()
        if n_pref >= n_folds:
            f_arr = _assign_prefix_balanced_folds(cdf, "inferred_subject_prefix", n_folds=n_folds)
            folds[cdf.index.to_numpy()] = f_arr
            strat = "inferred_subject_prefix_grouped"
        else:
            f_arr = _assign_contiguous_block_folds(cdf, ["inferred_sequence_id", "slice_index", "image_path"], n_folds=n_folds)
            folds[cdf.index.to_numpy()] = f_arr
            strat = f"contiguous_slice_block_{n_pref}prefix"
        for i in cdf.index:
            strategies[i] = strat
    df["cv_fold"] = folds
    df["cv_strategy"] = strategies
    df["patient_independent"] = False
    return df


def validate_cv_dataframe(
    df: pd.DataFrame,
    track: str,
    class_order: list[str],
    group_col: str | None = None,
    n_folds: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate fold completeness, class coverage across train_inner/val_inner/test_outer, and group overlap."""
    if (df["cv_fold"] < 0).any() or (df["cv_fold"] >= n_folds).any():
        raise ValueError(f"[{track}] Unassigned or out-of-range cv_fold values detected.")

    fold_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for k in range(n_folds):
        test_df = df[df["cv_fold"] == k]
        val_fold = (k + 1) % n_folds
        val_df = df[df["cv_fold"] == val_fold]
        train_df = df[~df["cv_fold"].isin([k, val_fold])]

        for cls in class_order:
            n_te = int((test_df["label"] == cls).sum())
            n_va = int((val_df["label"] == cls).sum())
            n_tr = int((train_df["label"] == cls).sum())
            fold_rows.append({
                "track": track,
                "fold": k + 1,
                "class": cls,
                "test_count": n_te,
                "val_count": n_va,
                "train_count": n_tr,
            })
            if n_te < 1 or n_va < 1 or n_tr < 1:
                errors.append(f"[{track} fold {k+1}] Missing class '{cls}' (tr={n_tr}, va={n_va}, te={n_te}).")

        # Sample overlap check
        s_tr, s_va, s_te = set(train_df["sample_id"]), set(val_df["sample_id"]), set(test_df["sample_id"])
        if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te):
            errors.append(f"[{track} fold {k+1}] Sample overlap across partitions!")

        # Check strict group leakage where applicable
        if group_col and group_col in df.columns:
            # For fundus check all classes; for OCT check multi-prefix classes only
            sub_df = df if track == "fundus_3class" else df[df["cv_strategy"].str.contains("prefix_grouped", na=False)]
            g_tr = set(sub_df.loc[sub_df.index.isin(train_df.index), group_col])
            g_va = set(sub_df.loc[sub_df.index.isin(val_df.index), group_col])
            g_te = set(sub_df.loc[sub_df.index.isin(test_df.index), group_col])
            overlap_cnt = len(g_tr & g_te) + len(g_va & g_te) + len(g_tr & g_va)
            if overlap_cnt > 0:
                errors.append(f"[{track} fold {k+1}] Group leakage detected on {group_col} ({overlap_cnt} overlapping groups).")

        inner_rows.append({
            "track": track,
            "fold": k + 1,
            "val_source_fold": val_fold + 1,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "unique_groups_total": int(df[group_col].nunique()) if group_col and group_col in df.columns else int(len(df)),
        })

    if errors:
        raise RuntimeError("Invalid CV split configuration detected:\n" + "\n".join(errors))

    diag = {
        "track": track,
        "n_samples": int(len(df)),
        "n_folds": n_folds,
        "strategies_used": sorted(df["cv_strategy"].unique().tolist()),
        "patient_independent": False,
        "validation_passed": True,
    }
    return diag, fold_rows, inner_rows


def build_and_validate_cv_splits(
    cfg: dict[str, Any] | None = None,
    cv_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate, validate, and export 5-fold CV assignments for all 4 tracks before training."""
    cfg = cfg or load_all_configs()
    cv_raw = cv_cfg or load_config("configs/cross_validation.yaml")
    cv_p = cv_raw.get("cross_validation", cv_raw)

    n_folds = int(cv_p["n_folds"])
    seed = int(cv_p["seed"])
    out_splits = resolve_path(cv_p["paths"]["splits_dir"])
    out_splits.mkdir(parents=True, exist_ok=True)

    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    fundus_raw = pd.read_csv(splits_dir / "fundus_splits.csv")
    oct3_raw = pd.read_csv(splits_dir / "oct_3class_splits.csv")
    octsev_raw = pd.read_csv(splits_dir / "oct_severity_splits.csv")

    f_cv = generate_fundus_cv_splits(fundus_raw, n_folds=n_folds, seed=seed)
    o3_cv = generate_oct_individual_cv_splits(oct3_raw, n_folds=n_folds)
    oseq_cv = generate_oct_sequence_cv_splits(
        oct3_raw,
        seq_len=int(cfg["oct"].get("sequence_length", 8)),
        stride=int(cfg["oct"].get("sequence_stride", 8)),
        min_seq_len=2,
        n_folds=n_folds,
    )
    osev_cv = generate_oct_severity_cv_splits(octsev_raw, n_folds=n_folds)

    f_cv.to_csv(out_splits / "fundus_3class_cv_folds.csv", index=False)
    o3_cv.to_csv(out_splits / "oct_individual_cv_folds.csv", index=False)
    oseq_cv.to_csv(out_splits / "oct_sequence_cv_folds.csv", index=False)
    osev_cv.to_csv(out_splits / "oct_severity_cv_folds.csv", index=False)

    diags: dict[str, Any] = {}
    all_fold_rows: list[dict[str, Any]] = []
    all_inner_rows: list[dict[str, Any]] = []

    for track_name, df_trk, cls_order, grp_col in (
        ("fundus_3class", f_cv, FUNDUS_CLASSES, "cluster_id"),
        ("oct_individual", o3_cv, OCT_3CLASSES, "inferred_subject_prefix"),
        ("oct_sequence", oseq_cv, OCT_3CLASSES, "inferred_subject_prefix"),
        ("oct_severity", osev_cv, OCT_SEVERITY_CLASSES, "inferred_subject_prefix"),
    ):
        d, frows, irows = validate_cv_dataframe(df_trk, track_name, cls_order, group_col=grp_col, n_folds=n_folds)
        diags[track_name] = d
        all_fold_rows.extend(frows)
        all_inner_rows.extend(irows)

    pd.DataFrame(all_fold_rows).to_csv(out_splits / "cv_fold_distributions.csv", index=False)
    pd.DataFrame(all_inner_rows).to_csv(out_splits / "cv_inner_split_summary.csv", index=False)
    (out_splits / "cv_leakage_diagnostics.json").write_text(json.dumps(diags, indent=2), encoding="utf-8")
    return {"diagnostics": diags, "splits_dir": str(out_splits)}

