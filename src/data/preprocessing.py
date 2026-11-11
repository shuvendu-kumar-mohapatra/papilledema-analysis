"""Image loading and modality-specific preprocessing (Fundus RGB/CLAHE & OCT Grayscale)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF


def apply_fundus_clahe(pil_img: Image.Image, clip_limit: float = 2.0, grid_size: tuple[int, int] = (8, 8)) -> Image.Image:
    """Apply contrast-limited adaptive histogram equalization on LAB L-channel."""
    rgb = np.asarray(pil_img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_eq = clahe.apply(l_ch)
    merged = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(merged)


def load_preprocessed_pil(
    path: str | Path,
    modality: str,
    image_size: int = 224,
    use_clahe: bool = False,
) -> Image.Image:
    """Load an image cleanly as RGB (fundus) or L grayscale (oct) and resize."""
    with Image.open(path) as im:
        if modality == "fundus":
            out = im.convert("RGB")
            if use_clahe:
                out = apply_fundus_clahe(out)
        elif modality == "oct":
            out = im.convert("L")
        else:
            raise ValueError(f"Unsupported modality '{modality}'")
        out = out.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return out


def pil_to_normalized_tensor(
    pil_img: Image.Image,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Convert PIL image to normalized float32 tensor (C, H, W)."""
    tensor = TF.to_tensor(pil_img)
    return TF.normalize(tensor, mean=list(mean), std=list(std))

