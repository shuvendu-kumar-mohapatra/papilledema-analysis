"""Comprehensive documentation-ready table generation matching project Results.docx format."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import resolve_path


def generate_all_documentation_tables(
    output_dir: str | Path = "results/tables/documentation_tables",
) -> dict[str, Path]:
    """Generate all documentation-ready tables in CSV format."""
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # -------------------------------------------------------------
    # SECTION A: DATASET & PREPROCESSING TABLES
    # -------------------------------------------------------------
    # A01: Original Dataset Distribution
    a01 = pd.DataFrame([
        {"Track": "EXP-01: Fundus 3-Class", "Normal": 778, "Papilledema": 295, "Pseudopapilledema": 294, "Total": 1367},
        {"Track": "EXP-02: Individual OCT", "Normal": 132, "Papilledema": 895, "Pseudopapilledema": 113, "Total": 1140},
        {"Track": "EXP-03: OCT Sequence", "Normal": "17 windows (2 seqs)", "Papilledema": "114 windows (17 seqs)", "Pseudopapilledema": "15 windows (2 seqs)", "Total": "146 windows (21 seqs)"},
        {"Track": "EXP-04: OCT Severity", "Normal": "G1: 40", "Papilledema": "G2: 598, G3: 184", "Pseudopapilledema": "G4: 73", "Total": "895 scans"},
    ])
    p = out_dir / "A01_original_dataset_distribution.csv"
    a01.to_csv(p, index=False)
    generated["A01"] = p

    # A02: Processed Dataset Distribution
    a02 = pd.DataFrame([
        {"Dataset": "Fundus", "Original Dimensions": "240 x 240 (dominant)", "Processed Dimensions": "224 x 224", "Raw Images": 1369, "Excluded": 2, "Retained": 1367, "Exclusion Reason": "Cross-class duplicate pair (Normal vs Pseudo)"},
        {"Dataset": "Individual OCT", "Original Dimensions": "381 x 596 (dominant)", "Processed Dimensions": "224 x 224", "Raw Images": 1141, "Excluded": 1, "Retained": 1140, "Exclusion Reason": "IR/SLO localizer reflectance image (heshama_001.jpg)"},
        {"Dataset": "OCT Sequence", "Original Dimensions": "Variable slices (381 x 596)", "Processed Dimensions": "224 x 224 per slice", "Raw Images": 1127, "Excluded": 0, "Retained": 1127, "Exclusion Reason": "L=8, stride=8, min_len=2 chunking into 146 windows"},
        {"Dataset": "OCT Severity", "Original Dimensions": "381 x 596 (dominant)", "Processed Dimensions": "224 x 224", "Raw Images": 895, "Excluded": 0, "Retained": 895, "Exclusion Reason": "Papilledema cohort with exploratory folder labels G1-G4"},
    ])
    p = out_dir / "A02_processed_dataset_distribution.csv"
    a02.to_csv(p, index=False)
    generated["A02"] = p

    # A03: Offline Augmentation Distribution
    sum_csv = resolve_path("results/augmentation/augmentation_summary/class_expansion_summary.csv")
    if sum_csv.exists():
        a03 = pd.read_csv(sum_csv)
    else:
        a03 = pd.DataFrame([
            {"dataset": "fundus_3class", "class_label": "Normal", "original_count": 778, "target_count": 778, "synthetic_generated": 0, "final_total": 778, "expansion_ratio": 1.0},
            {"dataset": "fundus_3class", "class_label": "Papilledema", "original_count": 295, "target_count": 750, "synthetic_generated": 455, "final_total": 750, "expansion_ratio": 2.542},
            {"dataset": "fundus_3class", "class_label": "Pseudopapilledema", "original_count": 294, "target_count": 750, "synthetic_generated": 456, "final_total": 750, "expansion_ratio": 2.551},
            {"dataset": "oct_individual", "class_label": "Normal", "original_count": 132, "target_count": 850, "synthetic_generated": 718, "final_total": 850, "expansion_ratio": 6.439},
            {"dataset": "oct_individual", "class_label": "Papilledema", "original_count": 895, "target_count": 895, "synthetic_generated": 0, "final_total": 895, "expansion_ratio": 1.0},
            {"dataset": "oct_individual", "class_label": "Pseudopapilledema", "original_count": 113, "target_count": 850, "synthetic_generated": 737, "final_total": 850, "expansion_ratio": 7.522},
            {"dataset": "oct_severity", "class_label": "G1", "original_count": 40, "target_count": 400, "synthetic_generated": 360, "final_total": 400, "expansion_ratio": 10.0},
            {"dataset": "oct_severity", "class_label": "G2", "original_count": 598, "target_count": 598, "synthetic_generated": 0, "final_total": 598, "expansion_ratio": 1.0},
            {"dataset": "oct_severity", "class_label": "G3", "original_count": 184, "target_count": 550, "synthetic_generated": 366, "final_total": 550, "expansion_ratio": 2.989},
            {"dataset": "oct_severity", "class_label": "G4", "original_count": 73, "target_count": 450, "synthetic_generated": 377, "final_total": 450, "expansion_ratio": 6.164},
        ])
    p = out_dir / "A03_offline_augmentation_distribution.csv"
    a03.to_csv(p, index=False)
    generated["A03"] = p

    # A04: Original vs Augmented Comparison
    a04_rows = []
    for (dname), grp in a03.groupby("dataset"):
        orig_tot = int(grp["original_count"].sum())
        syn_tot = int(grp["synthetic_generated"].sum())
        fin_tot = int(grp["final_total"].sum())
        a04_rows.append({
            "Experimental Track": dname,
            "Original Dataset Size": orig_tot,
            "Synthetic Images Generated": syn_tot,
            "Augmented Dataset Size": fin_tot,
            "Overall Expansion Ratio": round(fin_tot / orig_tot, 3),
            "Imbalance Ratio (Before)": round(grp["original_count"].max() / grp["original_count"].min(), 2),
            "Imbalance Ratio (After)": round(grp["final_total"].max() / grp["final_total"].min(), 2),
        })
    a04 = pd.DataFrame(a04_rows)
    p = out_dir / "A04_original_vs_augmented_dataset_comparison.csv"
    a04.to_csv(p, index=False)
    generated["A04"] = p

    # A06: Augmentation Configuration
    a06 = pd.DataFrame([
        {"Modality": "Fundus", "Transformation": "Horizontal Flip", "Parameter Range": "p = 0.5", "Clinical / Anatomical Justification": "OD/OS (right/left eye) bilateral symmetry is anatomically valid"},
        {"Modality": "Fundus", "Transformation": "Small Rotation", "Parameter Range": "[-12°, +12°]", "Clinical / Anatomical Justification": "Simulates natural patient head tilt and cyclotorsion"},
        {"Modality": "Fundus", "Transformation": "Controlled Scaling & Affine", "Parameter Range": "scale in [0.95, 1.05], trans in [-6, +6] px", "Clinical / Anatomical Justification": "Simulates slight camera-to-cornea working distance variation"},
        {"Modality": "Fundus", "Transformation": "Photometric Jitter", "Parameter Range": "brightness, contrast, saturation in [0.92, 1.08]", "Clinical / Anatomical Justification": "Simulates flash intensity and pupil dilation illumination differences"},
        {"Modality": "OCT B-scan", "Transformation": "Horizontal Flip", "Parameter Range": "p = 0.5", "Clinical / Anatomical Justification": "Temporal-to-nasal scan direction variation is anatomically valid"},
        {"Modality": "OCT B-scan", "Transformation": "Vertical Flip", "Parameter Range": "STRICTLY FORBIDDEN (p = 0.0)", "Clinical / Anatomical Justification": "Inversion destroys vitreoretinal top-bottom anatomical polarity"},
        {"Modality": "OCT B-scan", "Transformation": "Small Rotation", "Parameter Range": "[-6°, +6°] ([-4°, +4°] for severity)", "Clinical / Anatomical Justification": "Simulates scan angle tilt across optic nerve head"},
        {"Modality": "OCT B-scan", "Transformation": "Translation", "Parameter Range": "[-6, +6] px ([-4, +4] px for severity)", "Clinical / Anatomical Justification": "Simulates scan centration offsets"},
        {"Modality": "OCT B-scan", "Transformation": "Contrast & Brightness", "Parameter Range": "[0.92, 1.08] ([0.95, 1.05] for severity)", "Clinical / Anatomical Justification": "Simulates laser optical signal-to-noise ratio and speckle variability"},
    ])
    p = out_dir / "A06_augmentation_configuration.csv"
    a06.to_csv(p, index=False)
    generated["A06"] = p

    # -------------------------------------------------------------
    # SECTION B: FEATURE EXTRACTION TABLES
    # -------------------------------------------------------------
    b01 = pd.DataFrame([
        {"Feature Category": "Intensity Statistics", "Fundus Dimensions": 13, "OCT Dimensions": 13, "Description": "Mean, std, median, min, max, skewness, kurtosis, percentiles (5, 25, 75, 95), energy, entropy"},
        {"Feature Category": "Color Features (RGB)", "Fundus Dimensions": 38, "OCT Dimensions": 0, "Description": "12 statistics per R/G/B channel + R/G and R/B mean ratio metrics (omitted for grayscale OCT)"},
        {"Feature Category": "Texture: GLCM", "Fundus Dimensions": 12, "OCT Dimensions": 12, "Description": "Contrast, dissimilarity, homogeneity, energy, correlation, ASM across 0° and 90° angles"},
        {"Feature Category": "Texture: LBP", "Fundus Dimensions": 11, "OCT Dimensions": 11, "Description": "10-bin uniform LBP histogram + LBP entropy (P=8, R=1)"},
        {"Feature Category": "Structural & Edge", "Fundus Dimensions": 8, "OCT Dimensions": 8, "Description": "Sobel gradient mean/std/max, Canny edge pixel density, corner Harris count/mean"},
        {"Feature Category": "HOG Summary Stats", "Fundus Dimensions": 4, "OCT Dimensions": 4, "Description": "Mean, std, max, entropy of the HOG descriptor array"},
        {"Feature Category": "Total Scalar Features", "Fundus Dimensions": 86, "OCT Dimensions": 48, "Description": "Combined non-spatial scalar feature vector"},
        {"Feature Category": "HOG Descriptor Array", "Fundus Dimensions": 1296, "OCT Dimensions": 1296, "Description": "36 blocks x 36 values/block (pixels_per_cell=(32, 32), cells_per_block=(2, 2), 9 bins)"},
        {"Feature Category": "Total Handcrafted Features", "Fundus Dimensions": 1382, "OCT Dimensions": 1344, "Description": "Total handcrafted feature representation per image"},
    ])
    p = out_dir / "B01_feature_extraction_summary.csv"
    b01.to_csv(p, index=False)
    generated["B01"] = p

    # B03: HOG Configuration
    b03 = pd.DataFrame([
        {"Parameter": "Input Image Resolution", "Value": "224 x 224 pixels"},
        {"Parameter": "Color Space", "Value": "Grayscale (RGB converted to L for Fundus)"},
        {"Parameter": "Pixels per Cell", "Value": "(32, 32)"},
        {"Parameter": "Cells per Block", "Value": "(2, 2)"},
        {"Parameter": "Orientation Bins", "Value": "9 (unsigned 0° - 180°)"},
        {"Parameter": "Block Normalization", "Value": "L2-Hys (Lowe-style clipped L2 norm)"},
        {"Parameter": "Total Spatial Blocks", "Value": "6 x 6 = 36 blocks"},
        {"Parameter": "Values per Block", "Value": "2 x 2 x 9 = 36 values"},
        {"Parameter": "Total HOG Dimensions", "Value": "36 blocks x 36 values = 1,296 dimensions"},
    ])
    p = out_dir / "B03_hog_configuration.csv"
    b03.to_csv(p, index=False)
    generated["B03"] = p

    # -------------------------------------------------------------
    # SECTION C: CLASSIFICATION RESULT TABLES (CONDITION A)
    # -------------------------------------------------------------
    ev_dir = resolve_path("results/evaluation")
    if ev_dir.exists():
        for track_key, filename, out_name in [
            ("fundus_3class", "cv_summary_fundus_3class.csv", "C01_fundus_classification_results.csv"),
            ("oct_individual", "cv_summary_oct_individual.csv", "C02_oct_individual_classification_results.csv"),
            ("oct_sequence", "cv_summary_oct_sequence.csv", "C03_oct_sequence_window_level_results.csv"),
            ("oct_severity", "cv_summary_oct_severity.csv", "C05_oct_severity_grading_results.csv"),
        ]:
            src_f = ev_dir / filename
            if src_f.exists():
                df_c = pd.read_csv(src_f)
                dst_p = out_dir / out_name
                df_c.to_csv(dst_p, index=False)
                generated[out_name[:3]] = dst_p

        seq_lvl = ev_dir / "oct_sequence_level_summary.csv"
        if seq_lvl.exists():
            df_sl = pd.read_csv(seq_lvl)
            dst_p = out_dir / "C04_oct_sequence_level_results.csv"
            df_sl.to_csv(dst_p, index=False)
            generated["C04"] = dst_p

    # -------------------------------------------------------------
    # SECTION D: CROSS-VALIDATION COMPARISONS
    # -------------------------------------------------------------
    d01_p = out_dir / "D01_cross_validation_5fold.csv"
    if (ev_dir / "cv_model_summary.csv").exists():
        pd.read_csv(ev_dir / "cv_model_summary.csv").to_csv(d01_p, index=False)
        generated["D01"] = d01_p

    # -------------------------------------------------------------
    # SECTION G: ABLATION STUDY TABLES
    # -------------------------------------------------------------
    abl_dir = resolve_path("results/ablation_study")
    if abl_dir.exists():
        for src_name, out_name in [
            ("abl01_cnn_backbone_comparison.csv", "G01_ablation_cnn_backbone.csv"),
            ("abl02_rnn_lstm_vs_bilstm.csv", "G02_ablation_rnn_directionality.csv"),
            ("abl03_hybrid_cnn_rnn_comparison.csv", "G03_ablation_hybrid_architecture.csv"),
            ("abl04_backbone_trainability_folds.csv", "G04_ablation_backbone_trainability.csv"),
            ("abl05_oct_sequence_length_folds.csv", "G05_ablation_oct_sequence_length.csv"),
        ]:
            src_f = abl_dir / src_name
            if src_f.exists():
                dst_p = out_dir / out_name
                pd.read_csv(src_f).to_csv(dst_p, index=False)
                generated[out_name[:3]] = dst_p

    return generated


if __name__ == "__main__":
    out = generate_all_documentation_tables()
    print(f"Generated {len(out)} documentation tables in results/tables/documentation_tables/:")
    for k, v in out.items():
        print(f"  [{k}] -> {v.name}")
