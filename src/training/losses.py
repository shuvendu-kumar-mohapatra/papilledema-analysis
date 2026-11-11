"""Loss helpers including balanced class weighting for imbalanced medical classes."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Compute balanced inverse-frequency weights normalized to mean 1.0."""
    arr = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(arr) / (num_classes * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_fn(
    train_labels: Sequence[int],
    num_classes: int,
    use_class_weights: bool = True,
    device: torch.device | None = None,
) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss with optional balanced class weights."""
    if use_class_weights and len(train_labels) > 0:
        w = compute_class_weights(train_labels, num_classes)
        if device is not None:
            w = w.to(device)
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()

