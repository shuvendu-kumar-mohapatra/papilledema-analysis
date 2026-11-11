"""Baseline models: MLP and Autoencoder + Classifier (train-only representation learning)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """MLP operating on spatial pooled grid + global per-channel intensity statistics."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        pool_size: int = 16,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        in_dim = in_channels * pool_size * pool_size + 2 * in_channels
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev, hd), nn.GELU(), nn.Dropout(p=dropout)])
            prev = hd
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        ch_mean = x.mean(dim=(2, 3))
        ch_std = x.std(dim=(2, 3), unbiased=False)
        feats = torch.cat([pooled, ch_mean, ch_std], dim=1)
        return self.net(feats)


class AutoencoderClassifier(nn.Module):
    """Convolutional Autoencoder + linear/MLP classification head on latent code."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        latent_dim: int = 128,
        recon_size: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.recon_size = recon_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, latent_dim),
        )
        self.decoder_proj = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(recon_size, recon_size), mode="bilinear", align_corners=False),
            nn.Conv2d(16, in_channels, 3, padding=1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.decoder_proj(z)).view(z.size(0), 128, 4, 4)
        return self.decoder(h)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(x, size=(self.recon_size, self.recon_size), mode="bilinear", align_corners=False)
        recon = self.decode(self.encode(x))
        return F.mse_loss(recon, target)

    def forward_with_recon(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.head(z)
        target = F.interpolate(x, size=(self.recon_size, self.recon_size), mode="bilinear", align_corners=False)
        recon = self.decode(z)
        return logits, F.mse_loss(recon, target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

