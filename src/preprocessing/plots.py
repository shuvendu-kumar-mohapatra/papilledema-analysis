"""Publication-ready PNG & PDF plots for Phase 2 preprocessing comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data.augmentation import build_train_augmentations
from src.data.preprocessing import apply_fundus_clahe, load_preprocessed_pil
from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.utils.config import resolve_path


def _apply_pub_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def _save_pair(fig: plt.Figure, stem: Path) -> list[Path]:
    png_p = stem.with_suffix(".png")
    pdf_p = stem.with_suffix(".pdf")
    fig.savefig(png_p, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_p, bbox_inches="tight")
    plt.close(fig)
    return [png_p, pdf_p]


def _pick_representative_samples(df: pd.DataFrame, label_col: str, classes: list[str], n_per_class: int = 2) -> list[tuple[str, pd.Series]]:
    """Deterministically select representative samples per class (median-resolution + alternate)."""
    chosen: list[tuple[str, pd.Series]] = []
    for cls in classes:
        sub = df[df[label_col] == cls].sort_values("sample_id").reset_index(drop=True)
        if sub.empty:
            continue
        idxs = [0] if len(sub) == 1 else [0, len(sub) // 2]
        for i in idxs[:n_per_class]:
            chosen.append((cls, sub.iloc[i]))
    return chosen


def plot_class_distributions(cls_dist_df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    tracks = [
        ("fundus_3class", "EXP-01: Fundus 3-Class (N=1,367 images)", FUNDUS_CLASSES),
        ("oct_individual", "EXP-02: Individual OCT 3-Class (N=1,140 B-scans)", OCT_3CLASSES),
        ("oct_sequence", "EXP-03: OCT Sequences (N=146 whole-df windows)", OCT_3CLASSES),
        ("oct_severity", "EXP-04: OCT Severity Exploratory (N=895 B-scans)", OCT_SEVERITY_CLASSES),
    ]
    palette = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for ax, (trk, title, cls_order) in zip(axes.flat, tracks):
        sub = cls_dist_df[cls_dist_df["track"] == trk].set_index("class").reindex(cls_order).reset_index()
        bars = ax.bar(sub["class"], sub["count"], color=palette[: len(sub)], edgecolor="#222222", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel("Sample Count")
        ax.set_ylim(0, max(sub["count"]) * 1.22)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for bar, cnt, pct in zip(bars, sub["count"], sub["pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(sub["count"]) * 0.025,
                f"{int(cnt)}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    fig.suptitle("Class Distributions Across All Four Independent Experimental Tracks")
    fig.text(
        0.5,
        -0.01,
        "Note: OCT Normal/Pseudopapilledema stem from 1 prefix each; Severity G1–G4 labels are folder-derived (not clinically verified).",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _save_pair(fig, fig_dir / "fig01_dataset_class_distribution")


def plot_fundus_before_after(fundus_df: pd.DataFrame, image_size: int, fig_dir: Path) -> list[Path]:
    samples = _pick_representative_samples(fundus_df, "label", FUNDUS_CLASSES, n_per_class=2)
    aug_tf = build_train_augmentations("fundus", image_size)
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.5, 2.75 * n_rows))
    col_titles = [
        "1. Original Raw Image",
        f"2. Preprocessed (RGB {image_size}x{image_size})",
        "3. CLAHE LAB-L Variant (Ref.)",
        "4. Stochastic Train Aug.",
    ]
    torch.manual_seed(42)
    for r, (cls, rec) in enumerate(samples):
        p = resolve_path(rec["image_path"])
        with Image.open(p) as im:
            raw_rgb = im.convert("RGB")
            raw_sz = im.size
        prep_pil = load_preprocessed_pil(p, modality="fundus", image_size=image_size, use_clahe=False)
        clahe_pil = apply_fundus_clahe(prep_pil)
        aug_pil = aug_tf(prep_pil)

        imgs = [raw_rgb, prep_pil, clahe_pil, aug_pil]
        for c, (ax, im_show) in enumerate(zip(axes[r], imgs)):
            ax.imshow(im_show)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], pad=8)
            if c == 0:
                ax.set_ylabel(f"{cls}\n({raw_sz[0]}x{raw_sz[1]})", fontsize=9.5, fontweight="bold")
    fig.suptitle("Fundus Preprocessing Comparison Across All Three Diagnostic Classes")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_pair(fig, fig_dir / "fig02_fundus_before_after_grid")


def plot_oct_individual_before_after(oct_3_df: pd.DataFrame, image_size: int, fig_dir: Path) -> list[Path]:
    samples = _pick_representative_samples(oct_3_df, "label", OCT_3CLASSES, n_per_class=2)
    aug_tf = build_train_augmentations("oct", image_size)
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.5, 2.65 * n_rows))
    col_titles = [
        "1. Original Raw B-Scan",
        f"2. Grayscale L ({image_size}x{image_size})",
        "3. Standardized View ((x-0.5)/0.5)",
        "4. Train Aug. (No Vertical Flip)",
    ]
    torch.manual_seed(42)
    for r, (cls, rec) in enumerate(samples):
        p = resolve_path(rec["image_path"])
        with Image.open(p) as im:
            raw_mode = im.mode
            raw_sz = im.size
            raw_arr = np.asarray(im)
        prep_pil = load_preprocessed_pil(p, modality="oct", image_size=image_size, use_clahe=False)
        prep_arr = np.asarray(prep_pil, dtype=np.float32) / 255.0
        std_arr = (prep_arr - 0.5) / 0.5  # [-1, 1]
        aug_pil = aug_tf(prep_pil)

        axes[r, 0].imshow(raw_arr, cmap="gray" if raw_arr.ndim == 2 else None)
        axes[r, 1].imshow(prep_pil, cmap="gray", vmin=0, vmax=255)
        axes[r, 2].imshow(std_arr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axes[r, 3].imshow(aug_pil, cmap="gray", vmin=0, vmax=255)
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], pad=8)
        axes[r, 0].set_ylabel(f"{cls}\n({raw_mode}, {raw_sz[0]}x{raw_sz[1]})", fontsize=9, fontweight="bold")
    fig.suptitle("Individual OCT B-Scan Preprocessing Comparison (Normalized Range [-1, +1])")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_pair(fig, fig_dir / "fig03_oct_individual_before_after_grid")


def plot_oct_sequence_grid(oct_3_df: pd.DataFrame, seq_df: pd.DataFrame, image_size: int, fig_dir: Path) -> list[Path]:
    from src.data.datasets import OCTSequenceDataset

    ds = OCTSequenceDataset(oct_3_df, seq_len=8, stride=8, min_seq_len=2, image_size=image_size, is_train=False)
    # Pick 1 full 8-frame sequence per class + 1 partially padded window (actual_len < 8)
    selected_indices: list[int] = []
    for cls in OCT_3CLASSES:
        candidates = [i for i, item in enumerate(ds.sequences) if item["label"] == cls and len(item["paths"]) == 8]
        if not candidates:
            candidates = [i for i, item in enumerate(ds.sequences) if item["label"] == cls]
        if candidates:
            selected_indices.append(candidates[0])
    padded_candidates = [i for i, item in enumerate(ds.sequences) if len(item["paths"]) < 8]
    if padded_candidates:
        selected_indices.append(padded_candidates[0])

    n_rows = len(selected_indices)
    fig, axes = plt.subplots(n_rows, 8, figsize=(15.0, 2.25 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, idx in enumerate(selected_indices):
        item = ds.sequences[idx]
        (frames, mask), _ = ds[idx]
        for t in range(8):
            ax = axes[r, t]
            is_valid = bool(mask[t].item())
            img_01 = np.clip((frames[t, 0].numpy() * 0.5) + 0.5, 0.0, 1.0)
            ax.imshow(img_01, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Slice t={t+1}", fontsize=9.5)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor("#1b9e77" if is_valid else "#cc0000")
                sp.set_linewidth(1.8)
            if not is_valid:
                ax.text(image_size // 2, image_size // 2, "PADDED\n(mask=0)", color="#ff4444", ha="center", va="center", fontsize=8, fontweight="bold")
        row_tag = f"{item['label']}\n(valid={len(item['paths'])}/8)"
        axes[r, 0].set_ylabel(row_tag, fontsize=9, fontweight="bold")
    fig.suptitle("OCT Ordered Sequence Construction (T=8, Stride=8; Green=Valid Frame, Red=Zero-Padded Masked Frame)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_pair(fig, fig_dir / "fig04_oct_sequence_preprocessing_grid")


def plot_oct_severity_before_after(oct_sev_df: pd.DataFrame, image_size: int, fig_dir: Path) -> list[Path]:
    samples = _pick_representative_samples(oct_sev_df, "label", OCT_SEVERITY_CLASSES, n_per_class=1)
    fig, axes = plt.subplots(len(samples), 3, figsize=(10.5, 2.5 * len(samples)))
    col_titles = ["1. Original Raw B-Scan", f"2. Preprocessed L ({image_size}x{image_size})", "3. Intensity-Rescaled Standardized"]
    for r, (cls, rec) in enumerate(samples):
        p = resolve_path(rec["image_path"])
        with Image.open(p) as im:
            raw_sz = im.size
            raw_arr = np.asarray(im)
        prep_pil = load_preprocessed_pil(p, modality="oct", image_size=image_size, use_clahe=False)
        std_arr = (np.asarray(prep_pil, dtype=np.float32) / 255.0 - 0.5) / 0.5
        axes[r, 0].imshow(raw_arr, cmap="gray" if raw_arr.ndim == 2 else None)
        axes[r, 1].imshow(prep_pil, cmap="gray", vmin=0, vmax=255)
        axes[r, 2].imshow(std_arr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], pad=8)
        axes[r, 0].set_ylabel(f"{cls} (exploratory)\n({raw_sz[0]}x{raw_sz[1]})", fontsize=9, fontweight="bold")
    fig.suptitle("OCT Exploratory Severity Dataset (G1–G4 Folder-Derived Labels): Before vs. After Preprocessing")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_pair(fig, fig_dir / "fig05_oct_severity_before_after_grid")


def plot_resolution_analysis(f_metrics: pd.DataFrame, o_metrics: pd.DataFrame, image_size: int, fig_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    # 1. Raw resolution scatter
    f_vc = f_metrics.groupby(["raw_width", "raw_height"]).size().reset_index(name="n")
    o_vc = o_metrics.groupby(["raw_width", "raw_height"]).size().reset_index(name="n")
    axes[0].scatter(f_vc["raw_width"], f_vc["raw_height"], s=np.clip(f_vc["n"] * 1.2, 35, 350), alpha=0.75, color="#2b5c8f", edgecolor="#111111", label="Fundus raw")
    axes[0].scatter(o_vc["raw_width"], o_vc["raw_height"], s=np.clip(o_vc["n"] * 0.4, 35, 350), alpha=0.75, color="#d95f02", edgecolor="#111111", marker="^", label="OCT raw")
    axes[0].scatter([image_size], [image_size], s=180, color="#1b9e77", marker="*", edgecolor="#111111", zorder=5, label=f"Target ({image_size}x{image_size})")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Raw Image Resolutions vs. Target")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(frameon=True, fontsize=8.5)

    # 2. Aspect ratio boxplot
    axes[1].boxplot([f_metrics["raw_aspect_ratio"], o_metrics["raw_aspect_ratio"]], tick_labels=["Fundus (raw)", "OCT (raw)"], patch_artist=True, boxprops=dict(facecolor="#cfd8dc"))
    axes[1].axhline(1.0, color="#1b9e77", linestyle="--", linewidth=1.4, label="Target 1.0 (224x224)")
    axes[1].set_ylabel("Aspect Ratio (Width / Height)")
    axes[1].set_title("Aspect Ratio Before Standardization")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    axes[1].legend(fontsize=8.5)

    # 3. Color mode counts
    mode_counts = pd.concat([
        f_metrics["raw_mode"].value_counts().rename("Fundus"),
        o_metrics["raw_mode"].value_counts().rename("OCT"),
    ], axis=1).fillna(0)
    mode_counts.plot(kind="bar", ax=axes[2], color=["#2b5c8f", "#d95f02"], edgecolor="#222222")
    axes[2].set_title("Raw Color Modes (Before RGB / L Cast)")
    axes[2].set_ylabel("Image Count")
    axes[2].set_xlabel("PIL Color Mode")
    axes[2].tick_params(axis="x", rotation=0)
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Raw Image Dimensions, Aspect Ratios, and Color Modes Before vs. Target Standardization")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_pair(fig, fig_dir / "fig06_image_resolution_analysis")


def plot_intensity_distributions(f_hists: dict[str, np.ndarray], o_hists: dict[str, np.ndarray], f_metrics: pd.DataFrame, o_metrics: pd.DataFrame, fig_dir: Path) -> list[Path]:
    bins = np.arange(256)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))

    # Fundus RGB before vs after resize
    for ch, col, label in (("r", "#d62728", "R"), ("g", "#2ca02c", "G"), ("b", "#1f77b4", "B")):
        axes[0, 0].plot(bins, f_hists[f"raw_{ch}"], color=col, linestyle="-", linewidth=1.4, alpha=0.85, label=f"Raw {label}")
        axes[0, 0].plot(bins, f_hists[f"prep_{ch}"], color=col, linestyle="--", linewidth=1.4, alpha=0.85, label=f"Prep {label}")
    axes[0, 0].set_title("Fundus RGB Intensity Distributions (Raw vs. 224x224)")
    axes[0, 0].set_xlabel("Pixel Intensity (0–255)")
    axes[0, 0].set_ylabel("Normalized Density")
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 0].grid(alpha=0.25, linestyle="--")

    # Fundus luminance + CLAHE reference
    axes[0, 1].plot(bins, f_hists["raw_luma"], color="#333333", linewidth=1.7, label="Raw Luminance")
    axes[0, 1].plot(bins, f_hists["prep_luma"], color="#1f77b4", linestyle="--", linewidth=1.7, label="Preprocessed 224x224")
    axes[0, 1].plot(bins, f_hists["clahe_luma"], color="#d95f02", linewidth=1.7, label="CLAHE Reference")
    axes[0, 1].set_title("Fundus Luminance Histogram: Raw vs. Preprocessed vs. CLAHE")
    axes[0, 1].set_xlabel("Luminance Intensity (0–255)")
    axes[0, 1].set_ylabel("Normalized Density")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(alpha=0.25, linestyle="--")

    # OCT gray intensity before vs after
    axes[1, 0].plot(bins, o_hists["raw_gray"], color="#333333", linewidth=1.8, label="Raw OCT")
    axes[1, 0].plot(bins, o_hists["prep_gray"], color="#1b9e77", linestyle="--", linewidth=1.8, label="Preprocessed L (224x224)")
    axes[1, 0].set_title("OCT Grayscale Intensity Histogram (Raw vs. 224x224)")
    axes[1, 0].set_xlabel("Grayscale Intensity (0–255)")
    axes[1, 0].set_ylabel("Normalized Density")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(alpha=0.25, linestyle="--")

    # Post-normalization tensor means & stds
    axes[1, 1].hist(f_metrics["norm_tensor_mean"], bins=28, alpha=0.65, color="#2b5c8f", label=f"Fundus (mean={f_metrics['norm_tensor_mean'].mean():.2f})")
    axes[1, 1].hist(o_metrics["norm_tensor_mean"], bins=28, alpha=0.65, color="#d95f02", label=f"OCT (mean={o_metrics['norm_tensor_mean'].mean():.2f})")
    axes[1, 1].axvline(0.0, color="#222222", linestyle="--", linewidth=1.1)
    axes[1, 1].set_title("Per-Image Normalized Tensor Mean Distribution (Model Input)")
    axes[1, 1].set_xlabel("Normalized Tensor Mean")
    axes[1, 1].set_ylabel("Image Count")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(alpha=0.25, linestyle="--")

    fig.suptitle("Pixel Intensity & Normalization Analysis Before vs. After Pipeline Standardization")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_pair(fig, fig_dir / "fig07_pixel_intensity_distributions")


def plot_contrast_and_quality(before_after_df: pd.DataFrame, f_metrics: pd.DataFrame, o_metrics: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))

    # 1. RMS Contrast (All images per dataset)
    f_rms = [f_metrics["raw_rms_contrast"], f_metrics["prep_rms_contrast"], f_metrics["clahe_rms_contrast"]]
    o_rms = [o_metrics["raw_rms_contrast"], o_metrics["prep_rms_contrast"]]
    pos_f = [1, 2, 3]
    pos_o = [4.5, 5.5]
    bp1 = axes[0].boxplot(f_rms, positions=pos_f, widths=0.6, patch_artist=True, boxprops=dict(facecolor="#a6bddb"))
    bp2 = axes[0].boxplot(o_rms, positions=pos_o, widths=0.6, patch_artist=True, boxprops=dict(facecolor="#fdb462"))
    axes[0].set_xticks([1, 2, 3, 4.5, 5.5])
    axes[0].set_xticklabels(["Fundus\nRaw", "Fundus\nPrep", "Fundus\nCLAHE", "OCT\nRaw", "OCT\nPrep"], fontsize=8.5)
    axes[0].set_ylabel("RMS Contrast ([0,1] std)")
    axes[0].set_title("RMS Contrast Comparison")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    # 2. Sharpness (Laplacian Variance)
    f_lap = [np.log1p(f_metrics["raw_laplacian_var"]), np.log1p(f_metrics["prep_laplacian_var"]), np.log1p(f_metrics["clahe_laplacian_var"])]
    o_lap = [np.log1p(o_metrics["raw_laplacian_var"]), np.log1p(o_metrics["prep_laplacian_var"])]
    axes[1].boxplot(f_lap, positions=pos_f, widths=0.6, patch_artist=True, boxprops=dict(facecolor="#a6bddb"))
    axes[1].boxplot(o_lap, positions=pos_o, widths=0.6, patch_artist=True, boxprops=dict(facecolor="#fdb462"))
    axes[1].set_xticks([1, 2, 3, 4.5, 5.5])
    axes[1].set_xticklabels(["Fundus\nRaw", "Fundus\nPrep", "Fundus\nCLAHE", "OCT\nRaw", "OCT\nPrep"], fontsize=8.5)
    axes[1].set_ylabel("log(1 + Laplacian Variance)")
    axes[1].set_title("Edge Sharpness After Resizing")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    # 3. Gradient Energy per track (mean +/- std bar chart)
    sub_all = before_after_df[before_after_df["class"] == "ALL"].copy()
    labels = [f"{r.track}\n({r.stage})" for r in sub_all.itertuples(index=False)]
    vals = sub_all["sobel_grad_mean_mean"].to_numpy()
    errs = sub_all["sobel_grad_mean_std"].to_numpy()
    cols = ["#4e79a7" if s == "raw" else ("#59a14f" if s == "prep" else "#e15759") for s in sub_all["stage"]]
    x = np.arange(len(labels))
    axes[2].bar(x, vals, yerr=errs, capsize=3, color=cols, edgecolor="#222222", alpha=0.9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    axes[2].set_ylabel("Mean Sobel Gradient Magnitude")
    axes[2].set_title("Structural Gradient Energy by Stage")
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Contrast, Sharpness, and Structural Quality Metrics Before vs. After Preprocessing")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_pair(fig, fig_dir / "fig08_contrast_and_quality_analysis")


def generate_all_preprocessing_figures(
    cfg: dict[str, Any],
    fundus_df: pd.DataFrame,
    oct_3_df: pd.DataFrame,
    oct_sev_df: pd.DataFrame,
    f_metrics: pd.DataFrame,
    o_metrics: pd.DataFrame,
    f_hists: dict[str, np.ndarray],
    o_hists: dict[str, np.ndarray],
    seq_df: pd.DataFrame,
    cls_dist_df: pd.DataFrame,
    before_after_df: pd.DataFrame,
    fig_dir: Path,
) -> list[Path]:
    """Generate all 8 PNG+PDF preprocessing figures from real datasets and computed tables."""
    _apply_pub_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    f_size = int(cfg["fundus"]["image_size"])
    o_size = int(cfg["oct"]["image_size"])

    generated.extend(plot_class_distributions(cls_dist_df, fig_dir))
    generated.extend(plot_fundus_before_after(fundus_df, f_size, fig_dir))
    generated.extend(plot_oct_individual_before_after(oct_3_df, o_size, fig_dir))
    generated.extend(plot_oct_sequence_grid(oct_3_df, seq_df, o_size, fig_dir))
    generated.extend(plot_oct_severity_before_after(oct_sev_df, o_size, fig_dir))
    generated.extend(plot_resolution_analysis(f_metrics, o_metrics, o_size, fig_dir))
    generated.extend(plot_intensity_distributions(f_hists, o_hists, f_metrics, o_metrics, fig_dir))
    generated.extend(plot_contrast_and_quality(before_after_df, f_metrics, o_metrics, fig_dir))
    return generated

