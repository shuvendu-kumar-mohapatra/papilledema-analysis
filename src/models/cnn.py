"""CNN models (ResNet-50, DenseNet-121, EfficientNet-B0) with 1ch/3ch & freeze modes."""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Adapt an RGB 3-channel Conv2d to arbitrary in_channels (e.g. 1ch grayscale OCT)."""
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            mean_w = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _load_tv_model(name: str, pretrained: bool) -> nn.Module:
    """Load torchvision model with graceful offline fallback if weights are uncached."""
    builders = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    }
    if name not in builders:
        raise ValueError(f"Unsupported CNN backbone '{name}'. Expected one of {list(builders)}")
    fn, default_weights = builders[name]
    if not pretrained:
        return fn(weights=None)
    try:
        return fn(weights=default_weights)
    except Exception as exc:
        logger.warning("Pretrained weights unavailable for %s (%s); using random init.", name, exc)
        return fn(weights=None)


def build_cnn_backbone(
    name: str,
    in_channels: int = 3,
    pretrained: bool = True,
    backbone_mode: str = "full",
) -> tuple[nn.Module, int]:
    """Return a spatial feature extractor module (outputs (B, D, h, w)) and D."""
    base = _load_tv_model(name, pretrained=pretrained)

    if name == "resnet50":
        base.conv1 = _adapt_first_conv(base.conv1, in_channels)
        feat_dim = 2048
        spatial = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        partial_modules = [base.layer4]
    elif name == "densenet121":
        base.features.conv0 = _adapt_first_conv(base.features.conv0, in_channels)
        feat_dim = 1024
        spatial = nn.Sequential(base.features, nn.ReLU(inplace=False))
        partial_modules = [base.features.denseblock4, base.features.norm5]
    elif name == "efficientnet_b0":
        base.features[0][0] = _adapt_first_conv(base.features[0][0], in_channels)
        feat_dim = 1280
        spatial = base.features
        partial_modules = [base.features[-2], base.features[-1]]
    else:
        raise ValueError(f"Unsupported backbone '{name}'")

    if backbone_mode == "frozen":
        for p in spatial.parameters():
            p.requires_grad = False
    elif backbone_mode == "partial":
        for p in spatial.parameters():
            p.requires_grad = False
        for mod in partial_modules:
            for p in mod.parameters():
                p.requires_grad = True
    elif backbone_mode == "full":
        for p in spatial.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Invalid backbone_mode '{backbone_mode}'. Use 'frozen', 'partial', or 'full'.")

    return spatial, feat_dim


class CNNClassifier(nn.Module):
    """Standard 2D CNN classifier built on a shared spatial backbone."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = True,
        backbone_mode: str = "full",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.features, feat_dim = build_cnn_backbone(
            backbone_name, in_channels=in_channels, pretrained=pretrained, backbone_mode=backbone_mode
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.pool(feats)
        return self.classifier(pooled)

