"""High-performance RAM-resident shared memory dataset pipeline for cross-validation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.augmentation import build_eval_transform, build_train_augmentations
from src.data.preprocessing import apply_fundus_clahe, pil_to_normalized_tensor
from src.utils.config import resolve_path


CACHE_DIR = resolve_path("data/processed/ram_cache")


def ensure_ram_cache(track: str) -> Path:
    """Ensure pre-decoded, resized uint8 binary cache exists for the given track."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_bin = CACHE_DIR / f"{track}_224.bin"
    if cache_bin.exists() and cache_bin.stat().st_size > 0:
        return cache_bin

    splits_p = resolve_path(f"results/cross_validation/10_fold/splits/{track}_cv_folds.csv")
    if not splits_p.exists():
        raise FileNotFoundError(f"Cannot build RAM cache: missing splits file {splits_p}")

    df = pd.read_csv(splits_p)
    modality = "fundus" if "fundus" in track else "oct"

    if track == "oct_sequence":
        mmap = np.memmap(str(cache_bin), dtype=np.uint8, mode="w+", shape=(len(df), 8, 224, 224, 1))
        for i, row in df.iterrows():
            paths = json.loads(row["paths_json"])
            for t, p_rel in enumerate(paths[:8]):
                p = resolve_path(p_rel)
                with Image.open(p) as im:
                    im = im.convert("L").resize((224, 224), Image.BILINEAR)
                    mmap[i, t, :, :, 0] = np.array(im, dtype=np.uint8)
        mmap.flush()
    else:
        channels = 3 if modality == "fundus" else 1
        mmap = np.memmap(str(cache_bin), dtype=np.uint8, mode="w+", shape=(len(df), 224, 224, channels))
        use_clahe = modality == "fundus"
        for i, row in df.iterrows():
            p = resolve_path(row["image_path"])
            with Image.open(p) as im:
                if modality == "fundus":
                    im = im.convert("RGB")
                    if use_clahe:
                        im = apply_fundus_clahe(im)
                    im = im.resize((224, 224), Image.BILINEAR)
                    mmap[i] = np.array(im, dtype=np.uint8)
                else:
                    im = im.convert("L").resize((224, 224), Image.BILINEAR)
                    mmap[i, :, :, 0] = np.array(im, dtype=np.uint8)
        mmap.flush()

    return cache_bin


class RAMResidentImageDataset(Dataset):
    """Zero-copy memory-mapped dataset backed by RAM cache."""

    def __init__(
        self,
        df: pd.DataFrame,
        track: str,
        image_size: int = 224,
        is_train: bool = False,
        use_clahe: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        return_meta: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.track = track
        self.image_size = image_size
        self.is_train = is_train
        self.return_meta = return_meta
        self.modality = "fundus" if "fundus" in track else "oct"

        if mean is None:
            mean = [0.485, 0.456, 0.406] if self.modality == "fundus" else [0.5]
        if std is None:
            std = [0.229, 0.224, 0.225] if self.modality == "fundus" else [0.5]
        self.mean = mean
        self.std = std

        self.transform = build_train_augmentations(self.modality, image_size) if is_train else build_eval_transform(image_size)

        cache_bin = ensure_ram_cache(track)
        full_df = pd.read_csv(resolve_path(f"results/cross_validation/10_fold/splits/{track}_cv_folds.csv"))
        channels = 3 if self.modality == "fundus" else 1

        self.mmap = np.memmap(
            str(cache_bin),
            dtype=np.uint8,
            mode="r",
            shape=(len(full_df), image_size, image_size, channels),
        )

        path_to_idx = {p: i for i, p in enumerate(full_df["image_path"])}
        self.indices = [path_to_idx[p] for p in self.df["image_path"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        g_idx = self.indices[idx]
        raw_arr = self.mmap[g_idx]

        if self.modality == "fundus":
            im = Image.fromarray(raw_arr)
        else:
            im = Image.fromarray(raw_arr[:, :, 0])

        im = self.transform(im) if self.is_train else im
        x = pil_to_normalized_tensor(im, self.mean, self.std)
        row = self.df.iloc[idx]
        y = int(row["label_idx"])

        if not self.return_meta:
            return x, y

        group_col = "cluster_id" if "cluster_id" in row else "inferred_sequence_id"
        meta = {
            "sample_id": str(row["sample_id"]),
            "image_path": str(row["image_path"]),
            "label": str(row["label"]),
            "group_id": str(row.get(group_col, row["sample_id"])),
        }
        return x, y, meta


class RAMResidentOCTSequenceDataset(Dataset):
    """Zero-copy memory-mapped sequence dataset backed by RAM cache."""

    def __init__(
        self,
        df: pd.DataFrame,
        track: str = "oct_sequence",
        seq_len: int = 8,
        image_size: int = 224,
        is_train: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        return_meta: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.image_size = image_size
        self.is_train = is_train
        self.return_meta = return_meta
        self.mean = mean or [0.5]
        self.std = std or [0.5]

        self.transform = build_train_augmentations("oct", image_size) if is_train else build_eval_transform(image_size)

        cache_bin = ensure_ram_cache(track)
        full_df = pd.read_csv(resolve_path(f"results/cross_validation/10_fold/splits/{track}_cv_folds.csv"))
        self.mmap = np.memmap(
            str(cache_bin),
            dtype=np.uint8,
            mode="r",
            shape=(len(full_df), seq_len, image_size, image_size, 1),
        )

        id_to_idx = {sid: i for i, sid in enumerate(full_df["sample_id"])}
        self.indices = [id_to_idx[sid] for sid in self.df["sample_id"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        g_idx = self.indices[idx]
        row = self.df.iloc[idx]
        actual_len = int(row.get("actual_len", self.seq_len))

        frames = torch.zeros((self.seq_len, 1, self.image_size, self.image_size), dtype=torch.float32)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)

        seq_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if self.is_train else 0
        seed_state = torch.random.get_rng_state() if self.is_train else None

        for t in range(min(self.seq_len, actual_len)):
            slice_arr = self.mmap[g_idx, t, :, :, 0]
            im = Image.fromarray(slice_arr)
            if self.is_train:
                torch.manual_seed(seq_seed)
            im = self.transform(im)
            frames[t] = pil_to_normalized_tensor(im, self.mean, self.std)
            mask[t] = True

        if seed_state is not None:
            torch.random.set_rng_state(seed_state)

        y = int(row["label_idx"])
        if not self.return_meta:
            return (frames, mask), y

        paths = json.loads(row["paths_json"]) if "paths_json" in row else [row.get("image_path", "")]
        meta = {
            "sample_id": str(row["sample_id"]),
            "image_path": paths[0] if paths else "",
            "label": str(row["label"]),
            "group_id": str(row["sequence_id"]),
            "seq_len": actual_len,
        }
        return (frames, mask), y, meta


def get_ram_cache_report() -> dict[str, Any]:
    """Report total RAM cache footprint on disk and memory."""
    files = list(CACHE_DIR.glob("*.bin"))
    total_bytes = sum(f.stat().st_size for f in files)
    details = {f.name: round(f.stat().st_size / (1024 ** 2), 2) for f in files}
    return {
        "cache_dir": str(CACHE_DIR),
        "total_files": len(files),
        "total_cache_size_mb": round(total_bytes / (1024 ** 2), 2),
        "files_mb": details,
    }
