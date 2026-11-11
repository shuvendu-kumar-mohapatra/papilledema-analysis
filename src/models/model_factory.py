"""Unified model factory and parameter accounting."""

from __future__ import annotations

from typing import Any
import torch.nn as nn

from src.models.baselines import AutoencoderClassifier, MLPBaseline
from src.models.cnn import CNNClassifier
from src.models.hybrid import HybridCNNRNN
from src.models.rnn import OrderedRNNClassifier

SUPPORTED_MODELS = (
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "lstm",
    "bilstm",
    "resnet50_lstm",
    "resnet50_bilstm",
    "efficientnet_b0_bilstm",
    "mlp",
    "autoencoder_classifier",
)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def build_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 3,
    image_size: int = 224,
    cfg: dict[str, Any] | None = None,
    **overrides: Any,
) -> nn.Module:
    """Instantiate any supported CNN, RNN, Hybrid, or Baseline architecture."""
    mcfg = {}
    if cfg and "models" in cfg and model_name in cfg["models"]:
        mcfg = dict(cfg["models"][model_name])
    mcfg.update(overrides)

    if model_name in ("resnet50", "densenet121", "efficientnet_b0"):
        return CNNClassifier(
            backbone_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=bool(mcfg.get("pretrained", True)),
            backbone_mode=str(mcfg.get("backbone_mode", "full")),
            dropout=float(mcfg.get("dropout", 0.3)),
        )

    if model_name in ("lstm", "bilstm"):
        return OrderedRNNClassifier(
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
            patch_size=int(mcfg.get("patch_size", 32)),
            token_dim=int(mcfg.get("token_dim", 128)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            num_layers=int(mcfg.get("num_layers", 2)),
            bidirectional=(model_name == "bilstm") or bool(mcfg.get("bidirectional", False)),
            dropout=float(mcfg.get("dropout", 0.3)),
        )

    if model_name in ("resnet50_lstm", "resnet50_bilstm", "efficientnet_b0_bilstm"):
        backbone = str(mcfg.get("backbone", model_name.rsplit("_", 1)[0]))
        bidir = model_name.endswith("_bilstm") or bool(mcfg.get("bidirectional", False))
        return HybridCNNRNN(
            backbone_name=backbone,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=bool(mcfg.get("pretrained", True)),
            backbone_mode=str(mcfg.get("backbone_mode", "full")),
            proj_dim=int(mcfg.get("proj_dim", 256)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            num_layers=int(mcfg.get("num_layers", 1)),
            bidirectional=bidir,
            dropout=float(mcfg.get("dropout", 0.3)),
        )

    if model_name == "mlp":
        return MLPBaseline(
            num_classes=num_classes,
            in_channels=in_channels,
            pool_size=int(mcfg.get("pool_size", 16)),
            hidden_dims=tuple(mcfg.get("hidden_dims", (256, 128))),
            dropout=float(mcfg.get("dropout", 0.3)),
        )

    if model_name == "autoencoder_classifier":
        return AutoencoderClassifier(
            num_classes=num_classes,
            in_channels=in_channels,
            latent_dim=int(mcfg.get("latent_dim", 128)),
            recon_size=int(mcfg.get("recon_size", 64)),
            dropout=float(mcfg.get("dropout", 0.3)),
        )

    raise ValueError(f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}")

