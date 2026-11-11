"""Dataset integrity checks, duplicate clustering, IR/SLO detection, and leakage audits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import resolve_path

FUNDUS_CLASSES = ["Normal", "Papilledema", "Pseudopapilledema"]
OCT_3CLASS_MAP = {
    "Normal": "Normal",
    "Papilledema": "Papilledema",
    "Pseudo papilledema": "Pseudopapilledema",
}
OCT_3CLASSES = ["Normal", "Papilledema", "Pseudopapilledema"]
OCT_SEVERITY_CLASSES = ["G1", "G2", "G3", "G4"]


def verify_image_integrity(path: Path) -> dict[str, Any]:
    """Verify readability of an image file and return basic metadata and hashes."""
    raw_bytes = path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
            mode = im.mode
            arr = np.asarray(im.convert("RGB"))
        if arr.size == 0 or width == 0 or height == 0:
            raise ValueError("Empty image dimensions")
        pixel_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()
        return {
            "is_valid": True,
            "error": "",
            "width": int(width),
            "height": int(height),
            "mode": mode,
            "file_sha256": file_sha256,
            "pixel_sha256": pixel_sha256,
        }
    except Exception as exc:
        return {
            "is_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "width": 0,
            "height": 0,
            "mode": "",
            "file_sha256": file_sha256,
            "pixel_sha256": "",
        }


def is_ir_slo_image(path: Path) -> tuple[bool, str]:
    """Detect non-B-scan IR/SLO fundus reflectance images inside OCT raw data.

    In the Heidelberg Spectralis OCT dataset, all valid B-scans include the colored
    device watermark (max RGB channel diff >= 100) and horizontal B-scan aspect;
    the IR/SLO image is pure grayscale (max channel diff <= 10) with tall narrow shape.
    """
    with Image.open(path) as im:
        w, h = im.size
        arr = np.asarray(im.convert("RGB"), dtype=np.int16)
    ch_diff = int(
        max(
            np.max(np.abs(arr[:, :, 0] - arr[:, :, 1])),
            np.max(np.abs(arr[:, :, 1] - arr[:, :, 2])),
        )
    )
    if ch_diff <= 10 and (w < 320 or h > 650):
        return True, f"IR/SLO reflectance image detected (size=({w},{h}), max_ch_diff={ch_diff})"
    return False, ""


def parse_oct_filename(path: Path, class_token: str) -> dict[str, Any]:
    """Infer sequence ID and slice index from an OCT filename.

    Clearly flagged as inferred (`is_verified_patient=False`), never treated as
    verified patient identities.
    """
    stem = path.stem.strip()
    lower = stem.lower()

    m_nag = re.match(r"^([a-z]+)_(\d+)\s*\((\d+)\)$", lower)
    m_paren = re.match(r"^([a-z]+)\s*\((\d+)\)$", lower)
    m_under = re.match(r"^([a-z]+)_(\d+)$", lower)
    m_cat = re.match(r"^([a-z]+)(\d+)$", lower)

    if m_nag:
        prefix, fmt, idx = m_nag.group(1), f"under{m_nag.group(2)}_paren", int(m_nag.group(3))
    elif m_paren:
        prefix, fmt, idx = m_paren.group(1), "paren", int(m_paren.group(2))
    elif m_under:
        prefix, fmt, idx = m_under.group(1), f"under{len(m_under.group(2))}", int(m_under.group(2))
    elif m_cat:
        prefix, fmt, idx = m_cat.group(1), f"cat{len(m_cat.group(2))}", int(m_cat.group(2))
    else:
        prefix, fmt, idx = re.sub(r"[^a-z]", "", lower) or lower, "unknown", 0

    seq_id = f"{class_token}__{prefix}__{fmt}"
    return {
        "inferred_subject_prefix": prefix,
        "inferred_series_format": fmt,
        "inferred_sequence_id": seq_id,
        "slice_index": idx,
        "sequence_id_source": "inferred_from_filename",
        "is_verified_patient": False,
    }


def _compute_fundus_shift_ncc_matrix(paths: list[Path]) -> np.ndarray:
    """Compute pairwise shift-tolerant green-channel high-pass correlation for Fundus."""
    hps = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"OpenCV failed to read image: {p}")
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        g = img[:, :, 1].astype(np.float32)
        hps.append(g - cv2.GaussianBlur(g, (0, 0), 3.0))
    H = np.stack(hps)
    n = len(paths)
    Q = H[:, 12:52, 12:52].reshape(n, -1)
    Q = (Q - Q.mean(axis=1, keepdims=True)) / (Q.std(axis=1, keepdims=True) + 1e-6)
    max_corr = np.full((n, n), -1.0, dtype=np.float32)
    for dy in range(-8, 9, 2):
        for dx in range(-8, 9, 2):
            S = H[:, 12 + dy : 52 + dy, 12 + dx : 52 + dx].reshape(n, -1)
            S = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-6)
            c = (Q @ S.T) / Q.shape[1]
            np.maximum(max_corr, c, out=max_corr)
    max_corr = np.maximum(max_corr, max_corr.T)
    np.fill_diagonal(max_corr, 1.0)
    return max_corr


def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = [find(i) for i in range(n)]
    remap = {r: idx for idx, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]


def audit_fundus_dataset(
    raw_root: str | Path = "data/raw/fundus",
    near_dup_threshold: float = 0.65,
    cross_conflict_threshold: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate Kim Fundus images, detect exact/near duplicates, and group clusters."""
    root = resolve_path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Fundus raw directory not found: {root}")

    records = []
    corrupted = []
    for cls in FUNDUS_CLASSES:
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing Fundus class directory: {cls_dir}")
        for f in sorted(cls_dir.iterdir()):
            if not f.is_file():
                continue
            info = verify_image_integrity(f)
            rel = str(f.relative_to(resolve_path("."))) if f.is_relative_to(resolve_path(".")) else str(f)
            row = {
                "sample_id": f"fundus_{cls}_{f.stem}",
                "image_path": rel.replace("\\", "/"),
                "label": cls,
                "label_idx": FUNDUS_CLASSES.index(cls),
                **info,
            }
            if info["is_valid"]:
                records.append(row)
            else:
                corrupted.append(row)

    df = pd.DataFrame(records)
    paths = [resolve_path(p) for p in df["image_path"]]
    ncc = _compute_fundus_shift_ncc_matrix(paths)

    dup_rows = []
    exclude_indices = set()
    within_edges: list[tuple[int, int]] = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(ncc[i, j])
            exact_pix = df.loc[i, "pixel_sha256"] == df.loc[j, "pixel_sha256"]
            same_cls = df.loc[i, "label"] == df.loc[j, "label"]
            if not same_cls and (exact_pix or sim >= cross_conflict_threshold):
                dup_rows.append({
                    "image_a": df.loc[i, "image_path"],
                    "label_a": df.loc[i, "label"],
                    "image_b": df.loc[j, "image_path"],
                    "label_b": df.loc[j, "label"],
                    "similarity_ncc": round(sim, 5),
                    "exact_pixel_match": exact_pix,
                    "relation": "cross_class_conflict",
                })
                exclude_indices.add(i)
                exclude_indices.add(j)
            elif same_cls and (exact_pix or sim >= near_dup_threshold):
                dup_rows.append({
                    "image_a": df.loc[i, "image_path"],
                    "label_a": df.loc[i, "label"],
                    "image_b": df.loc[j, "image_path"],
                    "label_b": df.loc[j, "label"],
                    "similarity_ncc": round(sim, 5),
                    "exact_pixel_match": exact_pix,
                    "relation": "exact_duplicate" if (exact_pix or sim >= 0.995) else "near_duplicate",
                })
                within_edges.append((i, j))

    # Build clusters on retained valid samples only
    kept_mask = [i not in exclude_indices for i in range(n)]
    excluded_df = df.iloc[sorted(exclude_indices)].copy() if exclude_indices else pd.DataFrame(columns=df.columns)
    if not excluded_df.empty:
        excluded_df["exclusion_reason"] = "cross_class_exact_or_near_exact_duplicate"
    if corrupted:
        corr_df = pd.DataFrame(corrupted)
        corr_df["exclusion_reason"] = "corrupted_image"
        excluded_df = pd.concat([excluded_df, corr_df], ignore_index=True)

    kept_indices = [i for i in range(n) if kept_mask[i]]
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(kept_indices)}
    kept_edges = [
        (old_to_new[a], old_to_new[b])
        for a, b in within_edges
        if a in old_to_new and b in old_to_new
    ]
    comp_ids = _connected_components(len(kept_indices), kept_edges)

    manifest_df = df.iloc[kept_indices].copy().reset_index(drop=True)
    manifest_df["cluster_id"] = [f"fundus_cluster_{cid:04d}" for cid in comp_ids]
    cluster_sizes = Counter(comp_ids)
    manifest_df["cluster_size"] = [cluster_sizes[cid] for cid in comp_ids]

    dup_df = pd.DataFrame(dup_rows)
    summary = {
        "raw_total": int(n + len(corrupted)),
        "corrupted_count": int(len(corrupted)),
        "excluded_cross_class_count": int(len(exclude_indices)),
        "retained_total": int(len(manifest_df)),
        "retained_class_counts": manifest_df["label"].value_counts().to_dict(),
        "raw_class_counts": df["label"].value_counts().to_dict(),
        "unique_clusters": int(len(set(comp_ids))),
        "multi_image_clusters": int(sum(1 for c in cluster_sizes.values() if c > 1)),
        "images_in_multi_clusters": int(sum(c for c in cluster_sizes.values() if c > 1)),
        "near_dup_pairs": int(len(dup_df[dup_df["relation"] != "cross_class_conflict"])) if not dup_df.empty else 0,
        "cross_class_conflicts": int(len(dup_df[dup_df["relation"] == "cross_class_conflict"])) if not dup_df.empty else 0,
        "dimensions": manifest_df.apply(lambda r: f"{r['width']}x{r['height']}", axis=1).value_counts().to_dict(),
    }
    return manifest_df, dup_df, excluded_df, summary


def audit_oct_dataset(
    raw_root: str | Path = "data/raw/OCT/Original Data/OCT",
    excluded_paths: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate Original OCT dataset, exclude IR/SLO and forbidden folders, infer sequences."""
    root = resolve_path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"OCT raw directory not found: {root}")

    # Safety check: verify we are reading the valid Original Data/OCT root, not excluded clones
    norm_root = str(root).replace("\\", "/").lower()
    for bad in ("original data/original data/oct", "oct data with patient id", "augmented data"):
        if bad in norm_root:
            raise RuntimeError(f"Forbidden OCT folder referenced: {root}")

    records = []
    excluded_rows = []

    def _scan_dir(folder: Path, label_3class: str, label_severity: str | None) -> None:
        if not folder.exists():
            raise FileNotFoundError(f"Missing OCT folder: {folder}")
        class_token = label_severity if label_severity else label_3class
        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            rel = str(f.relative_to(resolve_path("."))).replace("\\", "/")
            info = verify_image_integrity(f)
            if not info["is_valid"]:
                excluded_rows.append({"image_path": rel, "folder_class": class_token, "exclusion_reason": f"corrupted: {info['error']}"})
                continue
            is_slo, reason = is_ir_slo_image(f)
            if is_slo:
                excluded_rows.append({"image_path": rel, "folder_class": class_token, "exclusion_reason": reason})
                continue
            seq_meta = parse_oct_filename(f, class_token)
            records.append({
                "sample_id": f"oct_{class_token}_{f.stem}".replace(" ", "_"),
                "image_path": rel,
                "label_3class": label_3class,
                "label_3class_idx": OCT_3CLASSES.index(label_3class),
                "label_severity": label_severity if label_severity else "",
                "label_severity_idx": OCT_SEVERITY_CLASSES.index(label_severity) if label_severity else -1,
                **info,
                **seq_meta,
            })

    _scan_dir(root / "Normal", "Normal", None)
    _scan_dir(root / "Pseudo papilledema", "Pseudopapilledema", None)
    for grade in OCT_SEVERITY_CLASSES:
        _scan_dir(root / "Papilledema" / grade, "Papilledema", grade)

    manifest_df = pd.DataFrame(records).sort_values(
        ["label_3class", "label_severity", "inferred_sequence_id", "slice_index", "image_path"]
    ).reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded_rows)

    summary = {
        "raw_total_scanned": int(len(manifest_df) + len(excluded_df)),
        "excluded_count": int(len(excluded_df)),
        "excluded_files": excluded_df.to_dict(orient="records") if not excluded_df.empty else [],
        "retained_bscan_total": int(len(manifest_df)),
        "class_3_counts": manifest_df["label_3class"].value_counts().to_dict(),
        "severity_counts": manifest_df[manifest_df["label_severity"] != ""]["label_severity"].value_counts().to_dict(),
        "inferred_sequence_counts": manifest_df.groupby("label_3class")["inferred_sequence_id"].nunique().to_dict(),
        "inferred_prefix_counts": manifest_df.groupby("label_3class")["inferred_subject_prefix"].nunique().to_dict(),
        "severity_prefix_counts": manifest_df[manifest_df["label_severity"] != ""].groupby("label_severity")["inferred_subject_prefix"].nunique().to_dict(),
        "dimensions": manifest_df.apply(lambda r: f"{r['width']}x{r['height']}", axis=1).value_counts().to_dict(),
        "excluded_folders_enforced": excluded_paths or [
            "data/raw/OCT/Original Data/Original Data/OCT",
            "data/raw/OCT/Original Data/OCT Data with Patient ID",
            "data/raw/OCT/Augmented Data",
        ],
    }
    return manifest_df, excluded_df, summary


def validate_no_leakage(
    split_df: pd.DataFrame,
    task: str,
    label_col: str = "label",
    expected_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Run strict split leakage and label integrity checks; raise on critical leakage."""
    errors = []
    warnings = []

    splits = {s: split_df[split_df["split"] == s] for s in ("train", "val", "test")}
    for s, sdf in splits.items():
        if sdf.empty:
            errors.append(f"Split '{s}' is empty for task {task}.")

    # 1. Source image overlap
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[a]["image_path"]) & set(splits[b]["image_path"])
        if overlap:
            errors.append(f"Source-image overlap between {a} and {b}: {len(overlap)} images")

    # 2. Forbidden folder leakage
    forbidden = ("original data/original data/oct", "oct data with patient id", "augmented data", "heshama_001.jpg")
    bad_paths = [
        p for p in split_df["image_path"]
        if any(tok in p.lower().replace("\\", "/") for tok in forbidden)
    ]
    if bad_paths:
        errors.append(f"Forbidden/excluded paths found in {task} splits: {bad_paths[:5]}")

    # 3. Exact pixel duplicate overlap across splits
    if "pixel_sha256" in split_df.columns:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            pix_ov = set(splits[a]["pixel_sha256"]) & set(splits[b]["pixel_sha256"])
            if pix_ov:
                errors.append(f"Exact pixel duplicate across {a} and {b}: {len(pix_ov)} hashes")

    # 4. Fundus near-duplicate cluster overlap
    if "cluster_id" in split_df.columns:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            cl_ov = set(splits[a]["cluster_id"]) & set(splits[b]["cluster_id"])
            if cl_ov:
                errors.append(f"Near-duplicate cluster overlap between {a} and {b}: {len(cl_ov)} clusters")

    # 5. OCT inferred-sequence / prefix leakage check
    if "inferred_sequence_id" in split_df.columns:
        for cls, cdf in split_df.groupby(label_col):
            n_prefixes = cdf["inferred_subject_prefix"].nunique()
            seq_splits = cdf.groupby("inferred_sequence_id")["split"].nunique()
            split_seqs = seq_splits[seq_splits > 1].index.tolist()
            if n_prefixes >= 3:
                # Multi-subject class MUST keep sequences and prefixes disjoint across splits
                pref_splits = cdf.groupby("inferred_subject_prefix")["split"].nunique()
                leaked_prefs = pref_splits[pref_splits > 1].index.tolist()
                if leaked_prefs or split_seqs:
                    errors.append(
                        f"Multi-subject OCT class '{cls}' leaked inferred sequences/prefixes across splits: "
                        f"prefixes={leaked_prefs}, seqs={split_seqs}"
                    )
            else:
                if split_seqs:
                    warnings.append(
                        f"Single-subject OCT class '{cls}' (n_prefixes={n_prefixes}) uses contiguous spatial-block "
                        f"splitting across splits on sequences {split_seqs}; NOT patient-independent."
                    )

    # 6. Label mapping consistency
    if expected_classes is not None:
        observed = set(split_df[label_col].unique())
        unexpected = observed - set(expected_classes)
        if unexpected:
            errors.append(f"Unexpected labels in {task}: {sorted(unexpected)}")
        for s in ("train", "val", "test"):
            missing = set(expected_classes) - set(splits[s][label_col].unique())
            if missing:
                errors.append(f"Missing classes {sorted(missing)} in {task} split '{s}'")

    report = {
        "task": task,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "split_counts": {s: len(sdf) for s, sdf in splits.items()},
        "class_by_split": {
            s: sdf[label_col].value_counts().to_dict() for s, sdf in splits.items()
        },
    }
    if errors:
        raise RuntimeError(f"CRITICAL DATA LEAKAGE / INTEGRITY FAILURE in {task}:\n" + "\n".join(errors))
    return report


def run_full_data_verification(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run Fundus + OCT audits and persist manifests, duplicate reports, and audit JSON."""
    manifests_dir = resolve_path(cfg["paths"]["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)

    f_cfg = cfg["fundus"]
    f_manifest, f_dups, f_excl, f_summary = audit_fundus_dataset(
        raw_root=f_cfg["root"],
        near_dup_threshold=float(f_cfg.get("near_dup_threshold", 0.65)),
        cross_conflict_threshold=float(f_cfg.get("cross_class_conflict_threshold", 0.90)),
    )
    f_manifest.to_csv(manifests_dir / "fundus_manifest.csv", index=False)
    f_dups.to_csv(manifests_dir / "fundus_duplicates.csv", index=False)
    f_excl.to_csv(manifests_dir / "fundus_excluded.csv", index=False)

    o_cfg = cfg["oct"]
    o_manifest, o_excl, o_summary = audit_oct_dataset(
        raw_root=o_cfg["root"],
        excluded_paths=cfg["paths"].get("excluded_oct_paths"),
    )
    o_manifest.to_csv(manifests_dir / "oct_manifest.csv", index=False)
    o_excl.to_csv(manifests_dir / "oct_excluded.csv", index=False)

    audit = {"fundus": f_summary, "oct": o_summary}
    with (manifests_dir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit

