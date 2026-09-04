"""Compact seasonal PhysGlacierFormer for sparse glacier-wide observations."""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskedAttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class PhysGlacierFormerV2(nn.Module):
    """Predict winter and summer SMB whose sum is annual glacier-wide SMB."""

    def __init__(
        self,
        n_dynamic_features: int,
        n_static_features: int,
        n_hypsometry_features: int = 3,
        d_model: int = 32,
        n_heads: int = 4,
        n_encoder_layers: int = 1,
        ff_dim: int = 96,
        dropout: float = 0.2,
        **_: object,
    ) -> None:
        super().__init__()
        self.dynamic_embedding = nn.Sequential(
            nn.Linear(n_dynamic_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.month_embedding = nn.Embedding(12, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_encoder_layers, enable_nested_tensor=False
        )
        self.accumulation_pool = MaskedAttentionPool(d_model)
        self.ablation_pool = MaskedAttentionPool(d_model)

        self.static_encoder = nn.Sequential(
            nn.Linear(n_static_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.hypsometry_encoder = nn.Sequential(
            nn.Linear(n_hypsometry_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        self.winter_head = make_head()
        self.summer_head = make_head()

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
        x_hypsometry: torch.Tensor,
        month_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.dynamic_embedding(x_dynamic)
        x = x + self.month_embedding(month_ids - 1)
        x = self.encoder(x)

        accumulation_mask = (month_ids >= 10) | (month_ids <= 4)
        ablation_mask = (month_ids >= 5) & (month_ids <= 9)
        accumulation_state = self.accumulation_pool(x, accumulation_mask)
        ablation_state = self.ablation_pool(x, ablation_mask)
        static_state = self.static_encoder(x_static)

        area = x_hypsometry[:, :, 0].clamp_min(0.0)
        area = area / area.sum(dim=1, keepdim=True).clamp_min(1e-8)
        band_state = self.hypsometry_encoder(x_hypsometry)
        hypsometry_state = torch.sum(band_state * area.unsqueeze(-1), dim=1)

        winter = self.winter_head(
            torch.cat([accumulation_state, static_state, hypsometry_state], dim=-1)
        ).squeeze(-1)
        summer = self.summer_head(
            torch.cat([ablation_state, static_state, hypsometry_state], dim=-1)
        ).squeeze(-1)
        annual = winter + summer
        return annual, winter, summer
