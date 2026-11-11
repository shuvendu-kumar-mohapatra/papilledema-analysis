"""Hybrid CNN-RNN architectures supporting 2D spatial token sequences and 5D OCT sequences."""

from __future__ import annotations

from contextlib import nullcontext
import torch
import torch.nn as nn

from src.models.cnn import build_cnn_backbone


class HybridCNNRNN(nn.Module):
    """CNN spatial encoder paired with LSTM/BiLSTM sequence modeler."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = True,
        backbone_mode: str = "full",
        proj_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.bidirectional = bidirectional
        self.cnn, feat_dim = build_cnn_backbone(
            backbone_name, in_channels=in_channels, pretrained=pretrained, backbone_mode=backbone_mode
        )
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, proj_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.rnn = nn.LSTM(
            input_size=proj_dim,
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

    def _rnn_ctx(self, is_cuda: bool):
        if is_cuda and self.rnn.num_layers > 1 and self.rnn.dropout > 0:
            return torch.backends.cudnn.flags(enabled=False)
        return nullcontext()

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mask: torch.Tensor | None = None
        if isinstance(x, (tuple, list)):
            x, mask = x[0], x[1]

        if x.ndim == 4:
            # Spatial feature grid (B, D, h, w) -> ordered spatial token sequence (B, h*w, proj_dim)
            fmap = self.cnn(x)
            b, d, h, w = fmap.shape
            tokens = fmap.flatten(2).transpose(1, 2)  # (B, h*w, D)
            tokens = self.proj(tokens)
            seq_len = tokens.size(1)
            tokens = tokens + self.pos_embed[:, :seq_len, :]
            with self._rnn_ctx(x.is_cuda):
                out, _ = self.rnn(tokens)
            pooled = out.mean(dim=1)
        elif x.ndim == 5:
            # Ordered B-scan volume slices (B, T, C, H, W)
            b, t, c, h, w = x.shape
            if mask is not None:
                valid_flat = mask.reshape(-1)
                if not valid_flat.any():
                    raise ValueError("OCT sequence batch contains no valid frames.")
                fmap = self.cnn(x.view(b * t, c, h, w)[valid_flat])
                valid_feats = self.pool2d(fmap).flatten(1)
                flat_feats = valid_feats.new_zeros((b * t, valid_feats.size(-1)))
                flat_feats[valid_flat] = valid_feats
                slice_feats = flat_feats.view(b, t, -1)
            else:
                fmap = self.cnn(x.view(b * t, c, h, w))
                slice_feats = self.pool2d(fmap).view(b, t, -1)
            tokens = self.proj(slice_feats) + self.pos_embed[:, :t, :]
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

