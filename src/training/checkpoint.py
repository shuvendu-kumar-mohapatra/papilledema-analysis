"""Atomic checkpoint saving and safe loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically save a PyTorch checkpoint via temporary file replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return target


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint safely."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return torch.load(p, map_location=map_location, weights_only=False)

