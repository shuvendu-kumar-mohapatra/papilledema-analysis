"""Offline data augmentation pipeline for Fundus and OCT classification tracks.

Physically generates, saves, and logs synthetic images to balance minority classes
while preserving medical realism, tracking provenance, and preventing cross-fold leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms.functional as TF

from src.utils.config import resolve_path


def _apply_fundus_augmentation(img: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str, dict[str, Any]]:
    """Apply conservative, anatomically valid transformations to a fundus image."""
    applied: list[str] = []
    params: dict[str, Any] = {}

    # 1. Horizontal flip (OD/OS variation is anatomically sound)
    hflip = bool(rng.random() < 0.5)
    params["hflip"] = hflip
    if hflip:
        img = TF.hflip(img)
        applied.append("hflip")

    # 2. Small rotation (retinal disc angle variation: +/- 12 degrees)
    angle = float(rng.uniform(-12.0, 12.0))
    params["rotation_deg"] = round(angle, 2)
    img = TF.rotate(img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
    applied.append("rotation")

    # 3. Controlled scaling and subtle translation
    scale = float(rng.uniform(0.95, 1.05))
    tx = int(rng.integers(-6, 7))
    ty = int(rng.integers(-6, 7))
    params["scale"] = round(scale, 3)
    params["translate_px"] = [tx, ty]
    img = TF.affine(img, angle=0, translate=[tx, ty], scale=scale, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
    applied.append("affine")

    # 4. Subtle color/contrast variation (simulating camera exposure)
    bright_factor = float(rng.uniform(0.92, 1.08))
    contrast_factor = float(rng.uniform(0.92, 1.08))
    params["brightness_factor"] = round(bright_factor, 3)
    params["contrast_factor"] = round(contrast_factor, 3)
    img = TF.adjust_brightness(img, bright_factor)
    img = TF.adjust_contrast(img, contrast_factor)
    applied.append("photometric")

    return img, "+".join(applied), params


def _apply_oct_augmentation(img: Image.Image, rng: np.random.Generator, is_severity: bool = False) -> tuple[Image.Image, str, dict[str, Any]]:
    """Apply conservative transformations to an OCT B-scan preserving layer stratification."""
    applied: list[str] = []
    params: dict[str, Any] = {}

    # Strict constraint: NEVER vertical flip (vitreous must remain superior, choroid inferior)
    params["vflip"] = False

    # 1. Horizontal flip (temporal vs nasal scan acquisition)
    hflip = bool(rng.random() < 0.5)
    params["hflip"] = hflip
    if hflip:
        img = TF.hflip(img)
        applied.append("hflip")

    # 2. Small rotation (fovea tilt: +/- 6 degrees, +/- 4 for severity)
    max_deg = 4.0 if is_severity else 6.0
    angle = float(rng.uniform(-max_deg, max_deg))
    params["rotation_deg"] = round(angle, 2)
    img = TF.rotate(img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
    applied.append("rotation")

    # 3. Controlled translation (+/- 4%)
    max_trans = 4 if is_severity else 6
    tx = int(rng.integers(-max_trans, max_trans + 1))
    ty = int(rng.integers(-max_trans, max_trans + 1))
    params["translate_px"] = [tx, ty]
    img = TF.affine(img, angle=0, translate=[tx, ty], scale=1.0, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
    applied.append("affine")

    # 4. Subtle contrast/brightness (laser intensity & speckle balance)
    var_range = 0.05 if is_severity else 0.08
    bright_factor = float(rng.uniform(1.0 - var_range, 1.0 + var_range))
    contrast_factor = float(rng.uniform(1.0 - var_range, 1.0 + var_range))
    params["brightness_factor"] = round(bright_factor, 3)
    params["contrast_factor"] = round(contrast_factor, 3)
    img = TF.adjust_brightness(img, bright_factor)
    img = TF.adjust_contrast(img, contrast_factor)
    applied.append("photometric")

    return img, "+".join(applied), params


def generate_offline_augmented_dataset(
    df: pd.DataFrame,
    track: str,
    output_root: str | Path = "results/augmentation",
    target_counts: dict[str, int] | None = None,
    image_size: int = 224,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically generate and save offline synthetic images to achieve balanced target counts.

    Returns:
        (manifest_df, summary_df)
    """
    out_root = resolve_path(output_root)
    track_dir = out_root / track
    track_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    is_fundus = (track == "fundus_3class")
    is_sev = (track == "oct_severity")

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # 1. Log original images into manifest
    for r in df.itertuples(index=False):
        orig_id = str(r.sample_id)
        cls_lbl = str(r.label)
        grp = getattr(r, "cluster_id", getattr(r, "inferred_subject_prefix", "unknown"))
        manifest_rows.append({
            "image_id": orig_id,
            "original_image_id": orig_id,
            "dataset": track,
            "class_label": cls_lbl,
            "augmentation_type": "original",
            "augmentation_parameters": "{}",
            "acquisition_group": str(grp),
            "is_synthetic": False,
            "output_path": str(r.image_path),
            "parent_cv_fold": getattr(r, "cv_fold", -1),
        })

    # 2. Determine required synthetic allocations per class
    counts = df["label"].value_counts().to_dict()
    max_orig = max(counts.values())

    if target_counts is None:
        target_counts = {c: max_orig for c in counts}

    for cls_name, orig_count in counts.items():
        t_count = int(target_counts.get(cls_name, orig_count))
        n_needed = max(0, t_count - orig_count)
        cls_df = df[df["label"] == cls_name].reset_index(drop=True)
        cls_dir = track_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        n_generated = 0
        if n_needed > 0:
            # Distribute needed augmentations cyclically across original samples
            indices = np.arange(len(cls_df))
            rng.shuffle(indices)
            rep_factor = int(np.ceil(n_needed / len(cls_df)))

            aug_count_per_orig: dict[str, int] = {}
            for rep in range(rep_factor):
                for idx in indices:
                    if n_generated >= n_needed:
                        break
                    row = cls_df.iloc[idx]
                    orig_id = str(row.sample_id)
                    aug_idx = aug_count_per_orig.get(orig_id, 0) + 1
                    aug_count_per_orig[orig_id] = aug_idx

                    # Load and resize original source image
                    src_path = resolve_path(row.image_path)
                    with Image.open(src_path) as im:
                        im_conv = im.convert("RGB") if is_fundus else im.convert("L")
                        im_resized = im_conv.resize((image_size, image_size), Image.BILINEAR)

                    # Apply transform
                    if is_fundus:
                        aug_im, aug_type, aug_params = _apply_fundus_augmentation(im_resized, rng)
                    else:
                        aug_im, aug_type, aug_params = _apply_oct_augmentation(im_resized, rng, is_severity=is_sev)

                    # Save synthetic file
                    syn_stem = f"{orig_id}_aug_{aug_idx:03d}"
                    ext = ".jpg" if is_fundus else ".png"
                    out_file = cls_dir / f"{syn_stem}{ext}"
                    if is_fundus:
                        aug_im.save(out_file, quality=95)
                    else:
                        aug_im.save(out_file)

                    grp = getattr(row, "cluster_id", getattr(row, "inferred_subject_prefix", "unknown"))
                    rel_out = out_file.relative_to(resolve_path(".")).as_posix()
                    manifest_rows.append({
                        "image_id": syn_stem,
                        "original_image_id": orig_id,
                        "dataset": track,
                        "class_label": cls_name,
                        "augmentation_type": aug_type,
                        "augmentation_parameters": json.dumps(aug_params),
                        "acquisition_group": str(grp),
                        "is_synthetic": True,
                        "output_path": rel_out,
                        "parent_cv_fold": getattr(row, "cv_fold", -1),
                    })
                    n_generated += 1

        summary_rows.append({
            "dataset": track,
            "class_label": cls_name,
            "original_count": orig_count,
            "target_count": t_count,
            "synthetic_generated": n_generated,
            "final_total": orig_count + n_generated,
            "expansion_ratio": round((orig_count + n_generated) / orig_count, 3),
        })

    m_df = pd.DataFrame(manifest_rows)
    s_df = pd.DataFrame(summary_rows)
    return m_df, s_df
