"""Train-only anatomically justified data augmentation for Fundus and OCT."""

from __future__ import annotations

from torchvision import transforms as T


def build_train_augmentations(modality: str, image_size: int = 224) -> T.Compose:
    """Build modality-specific training augmentations (never applied to val/test)."""
    if modality == "fundus":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ])
    if modality == "oct":
        # Preserve retinal vertical orientation (vitreous top, choroid bottom):
        # horizontal flip is anatomically valid (OD/OS), vertical flip is forbidden.
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=7),
            T.RandomAffine(degrees=0, translate=(0.05, 0.04)),
            T.ColorJitter(brightness=0.12, contrast=0.12),
        ])
    raise ValueError(f"Unsupported modality '{modality}'")


def build_eval_transform(image_size: int = 224) -> T.Compose:
    """Deterministic resize-only transform for validation and test splits."""
    return T.Compose([T.Resize((image_size, image_size))])

