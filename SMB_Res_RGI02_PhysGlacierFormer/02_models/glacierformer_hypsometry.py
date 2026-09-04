"""GlacierFormer with an explicit RGI hypsometry branch."""
from __future__ import annotations

import torch
import torch.nn as nn

from registry import register


@register("glacierformer_hypsometry")
class GlacierFormerHypsometry(nn.Module):
    """Fuse monthly climate, static terrain, and elevation-band area distribution."""

    def __init__(
        self,
        n_dynamic_features: int = 15,
        n_static_features: int = 18,
        n_hypsometry_features: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.15,
        **kwargs,
    ):
        super().__init__()
        self.dynamic_embedding = nn.Sequential(
            nn.Linear(n_dynamic_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.month_embedding = nn.Embedding(12, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.static_mlp = nn.Sequential(
            nn.Linear(n_static_features, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.hypsometry_mlp = nn.Sequential(
            nn.Linear(n_hypsometry_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
        x_hypsometry: torch.Tensor,
    ) -> torch.Tensor:
        _, n_months, _ = x_dynamic.shape
        month_ids = torch.arange(n_months, device=x_dynamic.device)

        x = self.dynamic_embedding(x_dynamic)
        x = x + self.month_embedding(month_ids).unsqueeze(0)
        dynamic_state = self.encoder(x).mean(dim=1)

        static_state = self.static_mlp(x_static)

        band_area = x_hypsometry[:, :, 0].clamp_min(0.0)
        weights = band_area / band_area.sum(dim=1, keepdim=True).clamp_min(1e-8)
        band_state = self.hypsometry_mlp(x_hypsometry)
        hypsometry_state = (band_state * weights.unsqueeze(-1)).sum(dim=1)

        fused = torch.cat([dynamic_state, static_state, hypsometry_state], dim=-1)
        return self.head(fused).squeeze(-1)
