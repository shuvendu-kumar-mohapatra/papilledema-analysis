"""Standalone LSTM and BiLSTM classifiers operating on ordered patch/slice sequences."""

from __future__ import annotations

from contextlib import nullcontext
import torch
import torch.nn as nn


class OrderedRNNClassifier(nn.Module):
    """Standalone LSTM / BiLSTM over ordered spatial image patches (4D) or B-scan slices (5D)."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 32,
        token_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.bidirectional = bidirectional
        patch_flat_dim = in_channels * patch_size * patch_size
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_flat_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        # Lightweight spatial stem for 5D B-scan sequence inputs (no CNN backbone)
        self.slice_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        grid = max(1, image_size // patch_size)
        n_patches = max(max_seq_len, grid * grid)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.rnn = nn.LSTM(
            input_size=token_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, num_classes),
        )

    def _image_to_ordered_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> ordered raster scan patches (B, N_patches, C*P*P)
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            new_h = max(p, (h // p) * p)
            new_w = max(p, (w // p) * p)
            x = nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            _, _, h, w = x.shape
        gh, gw = h // p, w // p
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, gh, gw, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, gh * gw, c * p * p)
        return self.patch_proj(patches)

    def _rnn_ctx(self, is_cuda: bool):
        # cuDNN multi-layer LSTM with inter-layer dropout triggers 0xC0000409 on Windows DLL unload
        if is_cuda and self.rnn.num_layers > 1 and self.rnn.dropout > 0:
            return torch.backends.cudnn.flags(enabled=False)
        return nullcontext()

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mask: torch.Tensor | None = None
        if isinstance(x, (tuple, list)):
            x, mask = x[0], x[1]

        if x.ndim == 4:
            tokens = self._image_to_ordered_patches(x)
            seq_len = tokens.size(1)
            tokens = tokens + self.pos_embed[:, :seq_len, :]
            with self._rnn_ctx(x.is_cuda):
                out, _ = self.rnn(tokens)
            pooled = out.mean(dim=1)
        elif x.ndim == 5:
            b, t, c, h, w = x.shape
            if mask is not None:
                valid_flat = mask.reshape(-1)
                if not valid_flat.any():
                    raise ValueError("OCT sequence batch contains no valid frames.")
                valid_feats = self.slice_stem(x.view(b * t, c, h, w)[valid_flat])
                flat_tokens = valid_feats.new_zeros((b * t, valid_feats.size(-1)))
                flat_tokens[valid_flat] = valid_feats
                tokens = flat_tokens.view(b, t, -1)
            else:
                tokens = self.slice_stem(x.view(b * t, c, h, w)).view(b, t, -1)
            tokens = tokens + self.pos_embed[:, :t, :]
            if mask is not None:
                lengths = mask.sum(dim=1).clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    tokens, lengths, batch_first=True, enforce_sorted=False
                )
                with self._rnn_ctx(x.is_cuda):
                    packed_out, _ = self.rnn(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=t)
                m = mask.unsqueeze(-1).float()
                pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                with self._rnn_ctx(x.is_cuda):
                    out, _ = self.rnn(tokens)
                pooled = out.mean(dim=1)
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {tuple(x.shape)}")

        return self.head(pooled)

