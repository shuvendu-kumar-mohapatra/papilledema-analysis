"""Standalone handcrafted image feature extraction (intensity, RGB, GLCM, LBP, HOG, edges)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from src.data.preprocessing import load_preprocessed_pil
from src.utils.config import resolve_path

LBP_N_BINS = 10  # uniform LBP for P=8 produces values 0..9
RGB_HIST_BINS = 8  # compact per-channel normalized histogram


def _shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) else 0.0


def extract_single_handcrafted(
    image_path: str | Path,
    modality: str,
    image_size: int = 224,
) -> tuple[dict[str, float], np.ndarray]:
    """Extract interpretable scalar feature dict and full HOG vector for one image."""
    pil_img = load_preprocessed_pil(resolve_path(image_path), modality=modality, image_size=image_size, use_clahe=False)
    arr = np.asarray(pil_img, dtype=np.uint8)

    if modality == "fundus":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    gf = gray.astype(np.float64)
    flat = gf.ravel()
    feats: dict[str, float] = {}

    # A. Intensity statistics
    p10, p25, p50, p75, p90 = np.percentile(flat, [10.0, 25.0, 50.0, 75.0, 90.0])
    gray_hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=False)
    gray_probs = gray_hist.astype(np.float64) / max(1, gray_hist.sum())
    feats.update({
        "int_mean": float(np.mean(flat)),
        "int_std": float(np.std(flat)),
        "int_min": float(np.min(flat)),
        "int_max": float(np.max(flat)),
        "int_p10": float(p10),
        "int_p25": float(p25),
        "int_median": float(p50),
        "int_p75": float(p75),
        "int_p90": float(p90),
        "int_iqr": float(p75 - p25),
        "int_skew": float(stats.skew(flat)) if np.std(flat) > 1e-8 else 0.0,
        "int_kurtosis": float(stats.kurtosis(flat)) if np.std(flat) > 1e-8 else 0.0,
        "int_entropy": _shannon_entropy(gray_probs),
    })

    # B. RGB colour features (Fundus only)
    if modality == "fundus":
        rf = arr.astype(np.float64)
        for ch_i, ch_name in enumerate(("r", "g", "b")):
            ch_vals = rf[:, :, ch_i].ravel()
            feats[f"rgb_{ch_name}_mean"] = float(np.mean(ch_vals))
            feats[f"rgb_{ch_name}_std"] = float(np.std(ch_vals))
            feats[f"rgb_{ch_name}_p10"] = float(np.percentile(ch_vals, 10.0))
            feats[f"rgb_{ch_name}_p90"] = float(np.percentile(ch_vals, 90.0))
            ch_hist, _ = np.histogram(arr[:, :, ch_i], bins=RGB_HIST_BINS, range=(0, 256))
            ch_prob = ch_hist.astype(np.float64) / max(1, ch_hist.sum())
            for b_i, val in enumerate(ch_prob):
                feats[f"rgb_{ch_name}_hist_{b_i}"] = float(val)
        feats["rgb_rg_ratio"] = float(feats["rgb_r_mean"] / max(1.0, feats["rgb_g_mean"]))
        feats["rgb_rb_ratio"] = float(feats["rgb_r_mean"] / max(1.0, feats["rgb_b_mean"]))

    # C1. GLCM texture features (quantized to 32 levels across 4 angles at distances 1 and 3)
    quant = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        quant,
        distances=[1, 3],
        angles=[0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
        levels=32,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        mat = graycoprops(glcm, prop)  # (n_dist, n_angles)
        feats[f"glcm_{prop}_d1"] = float(np.mean(mat[0, :]))
        feats[f"glcm_{prop}_d3"] = float(np.mean(mat[1, :]))

    # C2. Uniform LBP texture features (P=8, R=1)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0, LBP_N_BINS))
    lbp_prob = lbp_hist.astype(np.float64) / max(1, lbp_hist.sum())
    for b_i, val in enumerate(lbp_prob):
        feats[f"lbp_u8_bin_{b_i}"] = float(val)
    feats["lbp_entropy"] = _shannon_entropy(lbp_prob)

    # D. HOG + Edge / Structural statistics
    hog_vec = hog(
        gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    feats["hog_mean"] = float(np.mean(hog_vec))
    feats["hog_std"] = float(np.std(hog_vec))
    feats["hog_max"] = float(np.max(hog_vec))
    feats["hog_l2_norm"] = float(np.linalg.norm(hog_vec))

    # Edge / Gradient / Center-vs-Periphery structure
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    edges = cv2.Canny(gray, 50, 150)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    h, w = gray.shape
    c_h0, c_h1 = h // 4, (3 * h) // 4
    c_w0, c_w1 = w // 4, (3 * w) // 4
    center_mean = float(np.mean(gf[c_h0:c_h1, c_w0:c_w1]))
    periph_mask = np.ones((h, w), dtype=bool)
    periph_mask[c_h0:c_h1, c_w0:c_w1] = False
    periph_mean = float(np.mean(gf[periph_mask]))

    feats.update({
        "edge_canny_density": float(np.mean(edges > 0)),
        "edge_sobel_mean": float(np.mean(gmag)),
        "edge_sobel_std": float(np.std(gmag)),
        "edge_sobel_p95": float(np.percentile(gmag, 95.0)),
        "edge_laplacian_var": float(np.var(lap)),
        "struct_center_intensity": center_mean,
        "struct_periph_intensity": periph_mean,
        "struct_center_periph_ratio": float(center_mean / max(1.0, periph_mean)),
    })
    return feats, hog_vec


def extract_handcrafted_dataset(
    df: pd.DataFrame,
    modality: str,
    image_size: int = 224,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract handcrafted feature table and aligned HOG vector matrix for a dataset."""
    records: list[dict[str, Any]] = []
    hog_rows: list[np.ndarray] = []
    for rec in df.itertuples(index=False):
        feats, hog_vec = extract_single_handcrafted(rec.image_path, modality=modality, image_size=image_size)
        meta: dict[str, Any] = {
            "sample_id": rec.sample_id,
            "image_path": rec.image_path,
            "label": rec.label,
            "split": getattr(rec, "split", "unknown"),
        }
        if hasattr(rec, "label_severity") and pd.notna(rec.label_severity):
            meta["label_severity"] = str(rec.label_severity)
        if hasattr(rec, "cluster_id") and pd.notna(rec.cluster_id):
            meta["cluster_id"] = str(rec.cluster_id)
        if hasattr(rec, "inferred_subject_prefix") and pd.notna(rec.inferred_subject_prefix):
            meta["inferred_subject_prefix"] = str(rec.inferred_subject_prefix)
        if hasattr(rec, "inferred_sequence_id") and pd.notna(rec.inferred_sequence_id):
            meta["inferred_sequence_id"] = str(rec.inferred_sequence_id)
        meta.update(feats)
        records.append(meta)
        hog_rows.append(hog_vec)

    feat_df = pd.DataFrame(records)
    hog_mat = np.vstack(hog_rows) if hog_rows else np.zeros((0, 0), dtype=np.float32)
    return feat_df, hog_mat

