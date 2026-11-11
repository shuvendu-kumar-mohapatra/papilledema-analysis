"""Reusable PyTorch Datasets and DataLoaders for Fundus, OCT B-scan, and OCT Sequence tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import build_eval_transform, build_train_augmentations
from src.data.preprocessing import apply_fundus_clahe, pil_to_normalized_tensor
from src.data.split_manager import load_split_df
from src.utils.config import resolve_path


class MedicalImageDataset(Dataset):
    """Shared manifest-backed image dataset for Fundus and individual OCT B-scans."""

    def __init__(
        self,
        df: pd.DataFrame,
        modality: str,
        image_size: int = 224,
        is_train: bool = False,
        use_clahe: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        return_meta: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.modality = modality
        self.image_size = image_size
        self.is_train = is_train
        self.use_clahe = use_clahe
        self.return_meta = return_meta
        if mean is None:
            mean = [0.485, 0.456, 0.406] if modality == "fundus" else [0.5]
        if std is None:
            std = [0.229, 0.224, 0.225] if modality == "fundus" else [0.5]
        self.mean = mean
        self.std = std
        self.transform = build_train_augmentations(modality, image_size) if is_train else build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        row = self.df.iloc[idx]
        img_path = resolve_path(row["image_path"])
        try:
            with Image.open(img_path) as im:
                if self.modality == "fundus":
                    img = im.convert("RGB")
                    if self.use_clahe:
                        img = apply_fundus_clahe(img)
                else:
                    img = im.convert("L")
                img = self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed loading image {img_path}: {type(exc).__name__}: {exc}") from exc

        x = pil_to_normalized_tensor(img, self.mean, self.std)
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


class OCTSequenceDataset(Dataset):
    """Ordered B-scan sequence dataset built from contiguous slices of inferred OCT sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 8,
        stride: int = 8,
        min_seq_len: int = 2,
        image_size: int = 224,
        is_train: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        return_meta: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.image_size = image_size
        self.is_train = is_train
        self.mean = mean or [0.5]
        self.std = std or [0.5]
        self.return_meta = return_meta
        self.transform = build_train_augmentations("oct", image_size) if is_train else build_eval_transform(image_size)

        self.sequences: list[dict[str, Any]] = []
        ordered = df.sort_values(["inferred_sequence_id", "slice_index", "image_path"])
        for seq_id, sdf in ordered.groupby("inferred_sequence_id", sort=False):
            paths = sdf["image_path"].tolist()
            label_idx = int(sdf["label_idx"].iloc[0])
            label_str = str(sdf["label"].iloc[0])
            slices_idx = sdf["slice_index"].tolist()
            # Chunk into non-overlapping ordered sub-volumes of length <= seq_len
            for start in range(0, len(paths), stride):
                chunk_paths = paths[start : start + seq_len]
                chunk_slices = slices_idx[start : start + seq_len]
                if len(chunk_paths) < min_seq_len:
                    if self.sequences and self.sequences[-1]["sequence_id"] == seq_id:
                        continue
                if len(chunk_paths) >= 1:
                    self.sequences.append({
                        "sample_id": f"{seq_id}__w{start:03d}",
                        "sequence_id": str(seq_id),
                        "paths": chunk_paths,
                        "slice_indices": chunk_slices,
                        "label_idx": label_idx,
                        "label": label_str,
                    })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Any:
        item = self.sequences[idx]
        actual_len = len(item["paths"])
        frames = torch.zeros((self.seq_len, 1, self.image_size, self.image_size), dtype=torch.float32)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)

        # Draw sequence seed first so RNG advances per sample, then lock it across slices t=0..T-1
        seq_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if self.is_train else 0
        seed_state = torch.random.get_rng_state() if self.is_train else None
        for t, rel_path in enumerate(item["paths"]):
            img_path = resolve_path(rel_path)
            with Image.open(img_path) as im:
                gray = im.convert("L")
                if self.is_train:
                    torch.manual_seed(seq_seed)
                gray = self.transform(gray)
            frames[t] = pil_to_normalized_tensor(gray, self.mean, self.std)
            mask[t] = True
        if seed_state is not None:
            torch.random.set_rng_state(seed_state)

        y = item["label_idx"]
        if not self.return_meta:
            return (frames, mask), y
        meta = {
            "sample_id": item["sample_id"],
            "image_path": item["paths"][0],
            "label": item["label"],
            "group_id": item["sequence_id"],
            "seq_len": actual_len,
        }
        return (frames, mask), y, meta


def create_dataloaders(
    cfg: dict[str, Any],
    task: str,
    mode: str = "image",
    batch_size: int | None = None,
    image_size: int | None = None,
    use_clahe: bool | None = None,
    return_meta_eval: bool = False,
    pin_memory: bool | None = None,
) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    """Build reproducible train/val/test DataLoaders for any supported task and mode."""
    split_df = load_split_df(cfg, task)
    is_fundus = task == "fundus_3class"
    ds_cfg = cfg["fundus"] if is_fundus else cfg["oct"]
    modality = ds_cfg["modality"]
    img_sz = int(image_size or ds_cfg.get("image_size", 224))
    bs = int(batch_size or (cfg["experiments"]["oct_sequence"].get("batch_size", 4) if mode == "sequence" else cfg["training"]["batch_size"]))
    num_workers = int(cfg["training"].get("num_workers", 0))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    clahe = bool(ds_cfg.get("use_clahe", False) if use_clahe is None else use_clahe)
    mean = ds_cfg.get("normalize_mean")
    std = ds_cfg.get("normalize_std")

    splits = {s: split_df[split_df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    loaders: dict[str, DataLoader] = {}
    for split_name, sdf in splits.items():
        is_train = split_name == "train"
        ret_meta = return_meta_eval and not is_train
        if mode == "sequence":
            if is_fundus:
                raise ValueError("Sequence mode is only supported for OCT tasks.")
            ds: Dataset = OCTSequenceDataset(
                sdf,
                seq_len=int(ds_cfg.get("sequence_length", 8)),
                stride=int(ds_cfg.get("sequence_stride", 8)),
                image_size=img_sz,
                is_train=is_train,
                mean=mean,
                std=std,
                return_meta=ret_meta,
            )
        elif mode == "image":
            ds = MedicalImageDataset(
                sdf,
                modality=modality,
                image_size=img_sz,
                is_train=is_train,
                use_clahe=clahe,
                mean=mean,
                std=std,
                return_meta=ret_meta,
            )
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

        loaders[split_name] = DataLoader(
            ds,
            batch_size=bs if is_train else max(1, bs),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return loaders, splits

