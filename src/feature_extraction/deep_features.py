"""Standalone deep GAP embedding extraction from existing trained CNN checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import MedicalImageDataset
from src.models.cnn import CNNClassifier
from src.utils.config import resolve_path
from src.utils.device import clear_gpu_memory, get_device

DEEP_BACKBONE_DIMS = {
    "resnet50": 2048,
    "densenet121": 1024,
    "efficientnet_b0": 1280,
}


def extract_cnn_embeddings(
    df: pd.DataFrame,
    modality: str,
    backbone_name: str,
    num_classes: int,
    checkpoint_path: str | Path,
    image_size: int,
    norm_mean: list[float],
    norm_std: list[float],
    batch_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an existing trained CNN checkpoint and extract global-average-pooled embeddings."""
    if backbone_name not in DEEP_BACKBONE_DIMS:
        raise ValueError(f"Unsupported deep backbone '{backbone_name}'. Expected {list(DEEP_BACKBONE_DIMS)}")

    in_channels = 3 if modality == "fundus" else 1
    ckpt_p = resolve_path(checkpoint_path)
    ckpt_loaded = ckpt_p.exists()

    model = CNNClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=not ckpt_loaded,
        backbone_mode="full",
        dropout=0.0,
    )
    if ckpt_loaded:
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)

    device = get_device(prefer_gpu=True)
    model = model.to(device)
    model.eval()

    ds = MedicalImageDataset(
        df=df,
        modality=modality,
        image_size=image_size,
        is_train=False,
        mean=norm_mean,
        std=norm_std,
        use_clahe=False,
        return_meta=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    chunks: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for inputs, _ in loader:
            x = inputs.to(device, non_blocking=use_amp)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fmap = model.features(x)
                pooled = model.pool(fmap).flatten(1)
            chunks.append(pooled.float().cpu().numpy())

    mat = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.zeros((0, DEEP_BACKBONE_DIMS[backbone_name]), dtype=np.float32)
    del model
    clear_gpu_memory()

    meta = {
        "backbone": backbone_name,
        "embedding_dim": int(mat.shape[1]),
        "n_samples": int(mat.shape[0]),
        "checkpoint_path": str(ckpt_p),
        "checkpoint_loaded": bool(ckpt_loaded),
    }
    return mat, meta
