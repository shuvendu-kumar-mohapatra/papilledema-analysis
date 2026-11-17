"""Quantitative before-and-after image preprocessing analysis across all four tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.datasets import OCTSequenceDataset
from src.data.preprocessing import apply_fundus_clahe, load_preprocessed_pil, pil_to_normalized_tensor
from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.utils.config import load_all_configs, resolve_path


def _compute_gray_and_quality_metrics(arr_u8: np.ndarray) -> dict[str, float]:
    """Compute intensity, contrast, gradient energy, and sharpness metrics from a uint8 image."""
    if arr_u8.ndim == 3:
        gray = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr_u8
    g = gray.astype(np.float64)
    mean_v = float(np.mean(g))
    std_v = float(np.std(g))
    min_v = float(np.min(g))
    max_v = float(np.max(g))
    p05, p50, p95 = np.percentile(g, [5.0, 50.0, 95.0])
    # Michelson & RMS contrast on [0, 1] scale
    g01 = g / 255.0
    rms_contrast = float(np.std(g01))
    gmin01, gmax01 = float(np.min(g01)), float(np.max(g01))
    michelson = float((gmax01 - gmin01) / (gmax01 + gmin01)) if (gmax01 + gmin01) > 1e-8 else 0.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_energy = float(np.mean(np.hypot(gx, gy)))
    snr_proxy = float(mean_v / (std_v + 1e-6))
    return {
        "intensity_mean": mean_v,
        "intensity_std": std_v,
        "intensity_min": min_v,
        "intensity_median": float(p50),
        "intensity_max": max_v,
        "intensity_p05": float(p05),
        "intensity_p95": float(p95),
        "rms_contrast": rms_contrast,
        "michelson_contrast": michelson,
        "laplacian_var": lap_var,
        "sobel_grad_mean": grad_energy,
        "snr_proxy": snr_proxy,
    }


def _compute_rgb_stats(rgb_u8: np.ndarray) -> dict[str, float]:
    """Compute per-channel RGB mean and std on uint8 RGB array."""
    f = rgb_u8.astype(np.float64)
    return {
        "r_mean": float(np.mean(f[:, :, 0])),
        "g_mean": float(np.mean(f[:, :, 1])),
        "b_mean": float(np.mean(f[:, :, 2])),
        "r_std": float(np.std(f[:, :, 0])),
        "g_std": float(np.std(f[:, :, 1])),
        "b_std": float(np.std(f[:, :, 2])),
    }


def _analyze_image_dataset(
    df: pd.DataFrame,
    modality: str,
    image_size: int,
    norm_mean: list[float],
    norm_std: list[float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute before/after image metrics and pooled pixel histograms for a dataset DataFrame."""
    rows: list[dict[str, Any]] = []
    hist_bins = np.arange(257, dtype=np.float64)
    if modality == "fundus":
        hists: dict[str, np.ndarray] = {
            "raw_r": np.zeros(256, dtype=np.float64),
            "raw_g": np.zeros(256, dtype=np.float64),
            "raw_b": np.zeros(256, dtype=np.float64),
            "prep_r": np.zeros(256, dtype=np.float64),
            "prep_g": np.zeros(256, dtype=np.float64),
            "prep_b": np.zeros(256, dtype=np.float64),
            "clahe_luma": np.zeros(256, dtype=np.float64),
            "raw_luma": np.zeros(256, dtype=np.float64),
            "prep_luma": np.zeros(256, dtype=np.float64),
        }
    else:
        hists = {
            "raw_gray": np.zeros(256, dtype=np.float64),
            "prep_gray": np.zeros(256, dtype=np.float64),
        }

    for rec in df.itertuples(index=False):
        img_path = resolve_path(rec.image_path)
        with Image.open(img_path) as im:
            raw_mode = im.mode
            raw_w, raw_h = im.size
            if modality == "fundus":
                raw_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            else:
                raw_arr = np.asarray(im.convert("L"), dtype=np.uint8)

        prep_pil = load_preprocessed_pil(img_path, modality=modality, image_size=image_size, use_clahe=False)
        prep_arr = np.asarray(prep_pil, dtype=np.uint8)
        norm_tensor = pil_to_normalized_tensor(prep_pil, mean=norm_mean, std=norm_std).numpy()

        raw_m = _compute_gray_and_quality_metrics(raw_arr)
        prep_m = _compute_gray_and_quality_metrics(prep_arr)

        row: dict[str, Any] = {
            "sample_id": rec.sample_id,
            "image_path": rec.image_path,
            "label": rec.label,
            "split": getattr(rec, "split", "unknown"),
            "raw_mode": raw_mode,
            "raw_width": int(raw_w),
            "raw_height": int(raw_h),
            "raw_aspect_ratio": float(raw_w / max(1, raw_h)),
            "prep_width": int(image_size),
            "prep_height": int(image_size),
            "norm_tensor_mean": float(np.mean(norm_tensor)),
            "norm_tensor_std": float(np.std(norm_tensor)),
            "norm_tensor_min": float(np.min(norm_tensor)),
            "norm_tensor_max": float(np.max(norm_tensor)),
        }
        if hasattr(rec, "label_severity") and pd.notna(rec.label_severity):
            row["label_severity"] = str(rec.label_severity)
        if hasattr(rec, "inferred_sequence_id") and pd.notna(rec.inferred_sequence_id):
            row["inferred_sequence_id"] = str(rec.inferred_sequence_id)
        if hasattr(rec, "cluster_id") and pd.notna(rec.cluster_id):
            row["cluster_id"] = str(rec.cluster_id)

        for k, v in raw_m.items():
            row[f"raw_{k}"] = v
        for k, v in prep_m.items():
            row[f"prep_{k}"] = v

        if modality == "fundus":
            for k, v in _compute_rgb_stats(raw_arr).items():
                row[f"raw_{k}"] = v
            for k, v in _compute_rgb_stats(prep_arr).items():
                row[f"prep_{k}"] = v
            clahe_pil = apply_fundus_clahe(prep_pil)
            clahe_arr = np.asarray(clahe_pil, dtype=np.uint8)
            for k, v in _compute_gray_and_quality_metrics(clahe_arr).items():
                row[f"clahe_{k}"] = v
            for ch_idx, ch_name in enumerate(("r", "g", "b")):
                hists[f"raw_{ch_name}"] += np.histogram(raw_arr[:, :, ch_idx], bins=hist_bins)[0]
                hists[f"prep_{ch_name}"] += np.histogram(prep_arr[:, :, ch_idx], bins=hist_bins)[0]
            hists["raw_luma"] += np.histogram(cv2.cvtColor(raw_arr, cv2.COLOR_RGB2GRAY), bins=hist_bins)[0]
            hists["prep_luma"] += np.histogram(cv2.cvtColor(prep_arr, cv2.COLOR_RGB2GRAY), bins=hist_bins)[0]
            hists["clahe_luma"] += np.histogram(cv2.cvtColor(clahe_arr, cv2.COLOR_RGB2GRAY), bins=hist_bins)[0]
        else:
            hists["raw_gray"] += np.histogram(raw_arr, bins=hist_bins)[0]
            hists["prep_gray"] += np.histogram(prep_arr, bins=hist_bins)[0]

        rows.append(row)

    # Normalize histograms to density
    norm_hists: dict[str, np.ndarray] = {}
    for k, arr in hists.items():
        s = float(np.sum(arr))
        norm_hists[k] = arr / s if s > 0 else arr
    return pd.DataFrame(rows), norm_hists


def _build_before_after_summary(fundus_df: pd.DataFrame, oct_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate before/after quantitative metrics by dataset track and class."""
    summary_rows: list[dict[str, Any]] = []
    metric_bases = [
        "intensity_mean",
        "intensity_std",
        "rms_contrast",
        "michelson_contrast",
        "laplacian_var",
        "sobel_grad_mean",
        "snr_proxy",
    ]
    specs = [
        ("fundus_3class", "fundus", fundus_df, "label", FUNDUS_CLASSES, True),
        ("oct_individual", "oct", oct_df, "label", OCT_3CLASSES, False),
        (
            "oct_severity",
            "oct",
            oct_df[oct_df["label_severity"].notna()].copy(),
            "label_severity",
            OCT_SEVERITY_CLASSES,
            False,
        ),
    ]
    for track_name, modality, df_sub, lbl_col, cls_order, has_clahe in specs:
        groups = [("ALL", df_sub)] + [(c, df_sub[df_sub[lbl_col] == c]) for c in cls_order]
        for cls_name, g in groups:
            if g.empty:
                continue
            stages = ["raw", "prep"] + (["clahe"] if has_clahe else [])
            for stage in stages:
                entry: dict[str, Any] = {
                    "track": track_name,
                    "modality": modality,
                    "class": cls_name,
                    "stage": stage,
                    "n_images": int(len(g)),
                    "norm_tensor_mean": float(g["norm_tensor_mean"].mean()) if stage == "prep" else np.nan,
                    "norm_tensor_std": float(g["norm_tensor_std"].mean()) if stage == "prep" else np.nan,
                }
                for mb in metric_bases:
                    col = f"{stage}_{mb}"
                    entry[f"{mb}_mean"] = float(g[col].mean())
                    entry[f"{mb}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
                summary_rows.append(entry)
    return pd.DataFrame(summary_rows)


def run_preprocessing_comparison(
    cfg: dict[str, Any] | None = None,
    output_dir: str | Path = "results/preprocessing",
) -> dict[str, Any]:
    """Run quantitative before-and-after preprocessing analysis and generate all CSVs and figures."""
    from src.preprocessing.plots import generate_all_preprocessing_figures

    cfg = cfg or load_all_configs()
    out_root = resolve_path(output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    manifests_dir = resolve_path(cfg["paths"]["manifests_dir"])

    fundus_df = pd.read_csv(splits_dir / "fundus_splits.csv")
    oct_3_df = pd.read_csv(splits_dir / "oct_3class_splits.csv")
    oct_sev_df = pd.read_csv(splits_dir / "oct_severity_splits.csv")

    f_cfg = cfg["fundus"]
    o_cfg = cfg["oct"]

    # 1. Per-image quantitative analysis (before vs after)
    f_metrics, f_hists = _analyze_image_dataset(
        fundus_df,
        modality="fundus",
        image_size=int(f_cfg["image_size"]),
        norm_mean=[float(x) for x in f_cfg["normalize_mean"]],
        norm_std=[float(x) for x in f_cfg["normalize_std"]],
    )
    o_metrics, o_hists = _analyze_image_dataset(
        oct_3_df,
        modality="oct",
        image_size=int(o_cfg["image_size"]),
        norm_mean=[float(x) for x in o_cfg["normalize_mean"]],
        norm_std=[float(x) for x in o_cfg["normalize_std"]],
    )

    f_metrics.to_csv(out_root / "per_image_metrics_fundus.csv", index=False)
    o_metrics.to_csv(out_root / "per_image_metrics_oct.csv", index=False)

    # 2. Before-and-after aggregated comparison summary
    before_after_df = _build_before_after_summary(f_metrics, o_metrics)
    before_after_df.to_csv(out_root / "preprocessing_before_after_summary.csv", index=False)

    # 3. Dataset overview & class distribution across all 4 tracks
    audit_path = manifests_dir / "dataset_audit.json"
    audit_json = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}

    seq_len = int(o_cfg.get("sequence_length", 8))
    seq_stride = int(o_cfg.get("sequence_stride", 8))
    seq_all = OCTSequenceDataset(oct_3_df, seq_len=seq_len, stride=seq_stride, min_seq_len=2, is_train=False)
    seq_split_counts = {}
    for sp in ("train", "val", "test"):
        sub_ds = OCTSequenceDataset(oct_3_df[oct_3_df["split"] == sp], seq_len=seq_len, stride=seq_stride, min_seq_len=2, is_train=False)
        seq_split_counts[sp] = sub_ds

    seq_rows: list[dict[str, Any]] = []
    for item in seq_all.sequences:
        seq_rows.append({
            "sample_id": item["sample_id"],
            "sequence_id": item["sequence_id"],
            "label": item["label"],
            "actual_len": int(len(item["paths"])),
            "padded_len": int(seq_len),
            "padded_frames": int(seq_len - len(item["paths"])),
            "valid_mask_ratio": float(len(item["paths"]) / seq_len),
            "start_slice_idx": int(item["slice_indices"][0]),
            "end_slice_idx": int(item["slice_indices"][-1]),
            "first_image_path": str(item["paths"][0]),
        })
    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(out_root / "oct_sequence_preprocessing_summary.csv", index=False)

    # Track-level class distribution table
    cls_rows: list[dict[str, Any]] = []
    for c in FUNDUS_CLASSES:
        n = int((fundus_df["label"] == c).sum())
        cls_rows.append({"track": "fundus_3class", "class": c, "count": n, "pct": round(100.0 * n / len(fundus_df), 2), "unit": "image", "notes": "1,367 retained (2 cross-class duplicates excluded)"})
    for c in OCT_3CLASSES:
        n = int((oct_3_df["label"] == c).sum())
        cls_rows.append({"track": "oct_individual", "class": c, "count": n, "pct": round(100.0 * n / len(oct_3_df), 2), "unit": "B-scan", "notes": "1,140 retained (1 IR/SLO image excluded)"})
    for c in OCT_3CLASSES:
        n = int((seq_df["label"] == c).sum())
        cls_rows.append({"track": "oct_sequence", "class": c, "count": n, "pct": round(100.0 * n / len(seq_df), 2), "unit": "window (8-slice max)", "notes": "146 whole-dataset windows (147 in fixed-split train/val/test)"})
    for c in OCT_SEVERITY_CLASSES:
        n = int((oct_sev_df["label"] == c).sum())
        cls_rows.append({"track": "oct_severity", "class": c, "count": n, "pct": round(100.0 * n / len(oct_sev_df), 2), "unit": "B-scan (exploratory)", "notes": "Folder-derived labels only; not clinically validated"})
    cls_dist_df = pd.DataFrame(cls_rows)
    cls_dist_df.to_csv(out_root / "class_distribution.csv", index=False)

    # Dataset overview summary table
    overview_rows = [
        {
            "track": "fundus_3class",
            "modality": "Fundus RGB",
            "raw_retained_samples": int(len(fundus_df)),
            "excluded_samples": int(audit_json.get("fundus", {}).get("excluded_cross_class_duplicates", 2)),
            "raw_color_modes": ",".join(sorted(f_metrics["raw_mode"].unique())),
            "raw_width_range": f"{int(f_metrics['raw_width'].min())}-{int(f_metrics['raw_width'].max())}",
            "raw_height_range": f"{int(f_metrics['raw_height'].min())}-{int(f_metrics['raw_height'].max())}",
            "target_size": f"{f_cfg['image_size']}x{f_cfg['image_size']}",
            "target_channels": int(f_cfg["in_channels"]),
            "use_clahe_default": bool(f_cfg.get("use_clahe", False)),
            "normalization_mean": str(f_cfg["normalize_mean"]),
            "normalization_std": str(f_cfg["normalize_std"]),
        },
        {
            "track": "oct_individual",
            "modality": "OCT Grayscale",
            "raw_retained_samples": int(len(oct_3_df)),
            "excluded_samples": int(audit_json.get("oct", {}).get("excluded_non_bscan", 1)),
            "raw_color_modes": ",".join(sorted(o_metrics["raw_mode"].unique())),
            "raw_width_range": f"{int(o_metrics['raw_width'].min())}-{int(o_metrics['raw_width'].max())}",
            "raw_height_range": f"{int(o_metrics['raw_height'].min())}-{int(o_metrics['raw_height'].max())}",
            "target_size": f"{o_cfg['image_size']}x{o_cfg['image_size']}",
            "target_channels": int(o_cfg["in_channels"]),
            "use_clahe_default": False,
            "normalization_mean": str(o_cfg["normalize_mean"]),
            "normalization_std": str(o_cfg["normalize_std"]),
        },
        {
            "track": "oct_sequence",
            "modality": "OCT Sequence (T<=8)",
            "raw_retained_samples": int(len(seq_df)),
            "excluded_samples": int(len(oct_3_df) - seq_df["actual_len"].sum()),
            "raw_color_modes": ",".join(sorted(o_metrics["raw_mode"].unique())),
            "raw_width_range": f"{int(o_metrics['raw_width'].min())}-{int(o_metrics['raw_width'].max())}",
            "raw_height_range": f"{int(o_metrics['raw_height'].min())}-{int(o_metrics['raw_height'].max())}",
            "target_size": f"{seq_len}x1x{o_cfg['image_size']}x{o_cfg['image_size']}",
            "target_channels": int(o_cfg["in_channels"]),
            "use_clahe_default": False,
            "normalization_mean": str(o_cfg["normalize_mean"]),
            "normalization_std": str(o_cfg["normalize_std"]),
        },
        {
            "track": "oct_severity",
            "modality": "OCT Grayscale (G1-G4 exploratory)",
            "raw_retained_samples": int(len(oct_sev_df)),
            "excluded_samples": 0,
            "raw_color_modes": ",".join(sorted(o_metrics.loc[o_metrics["label_severity"].notna(), "raw_mode"].unique())),
            "raw_width_range": f"{int(o_metrics.loc[o_metrics['label_severity'].notna(), 'raw_width'].min())}-{int(o_metrics.loc[o_metrics['label_severity'].notna(), 'raw_width'].max())}",
            "raw_height_range": f"{int(o_metrics.loc[o_metrics['label_severity'].notna(), 'raw_height'].min())}-{int(o_metrics.loc[o_metrics['label_severity'].notna(), 'raw_height'].max())}",
            "target_size": f"{o_cfg['image_size']}x{o_cfg['image_size']}",
            "target_channels": int(o_cfg["in_channels"]),
            "use_clahe_default": False,
            "normalization_mean": str(o_cfg["normalize_mean"]),
            "normalization_std": str(o_cfg["normalize_std"]),
        },
    ]
    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(out_root / "dataset_overview.csv", index=False)

    # Dimension distribution table
    dim_rows: list[dict[str, Any]] = []
    for ds_name, mdf in (("fundus", f_metrics), ("oct", o_metrics)):
        vc = mdf.groupby(["raw_width", "raw_height", "raw_mode"]).size().reset_index(name="count")
        vc.insert(0, "dataset", ds_name)
        vc["aspect_ratio"] = (vc["raw_width"] / vc["raw_height"]).round(4)
        dim_rows.append(vc)
    dim_df = pd.concat(dim_rows, ignore_index=True).sort_values(["dataset", "count"], ascending=[True, False])
    dim_df.to_csv(out_root / "image_dimensions_summary.csv", index=False)

    # Save pooled histograms as NPZ for fast reuse in reports/notebooks
    np.savez_compressed(
        out_root / "pixel_intensity_histograms.npz",
        bins=np.arange(256, dtype=np.int32),
        **{f"fundus_{k}": v for k, v in f_hists.items()},
        **{f"oct_{k}": v for k, v in o_hists.items()},
    )

    # 4. Generate publication-ready PNG + PDF figures
    fig_paths = generate_all_preprocessing_figures(
        cfg=cfg,
        fundus_df=fundus_df,
        oct_3_df=oct_3_df,
        oct_sev_df=oct_sev_df,
        f_metrics=f_metrics,
        o_metrics=o_metrics,
        f_hists=f_hists,
        o_hists=o_hists,
        seq_df=seq_df,
        cls_dist_df=cls_dist_df,
        before_after_df=before_after_df,
        fig_dir=fig_dir,
    )

    summary_manifest = {
        "output_dir": str(out_root),
        "figures_dir": str(fig_dir),
        "fundus_images_analyzed": int(len(f_metrics)),
        "oct_images_analyzed": int(len(o_metrics)),
        "oct_sequence_windows_whole": int(len(seq_df)),
        "oct_sequence_slices_used_whole": int(seq_df["actual_len"].sum()),
        "oct_sequence_windows_fixed_split": {k: len(v) for k, v in seq_split_counts.items()},
        "figures_generated": [str(p) for p in fig_paths],
    }
    (out_root / "preprocessing_manifest.json").write_text(json.dumps(summary_manifest, indent=2), encoding="utf-8")
    return summary_manifest

