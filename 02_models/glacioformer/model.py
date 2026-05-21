# 02_models/glacioformer/model.py
"""
GlacioFormer v1 — encoder-only dual-branch Transformer for annual SMB.
Adapted from SMB_Res_Glacierformer_Byclaude/02_model/models/v1_transformer.py.
Registered as 'glacioformer' in model registry.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import torch.nn as nn
from registry import register

@register('glacioformer')
class GlacioFormer(nn.Module):
    name = 'glacioformer'

    def __init__(self, n_dynamic_features=15, n_static_features=10,
                 d_model=64, n_heads=4, n_encoder_layers=2,
                 ff_dim=256, dropout=0.15, **kwargs):
        super().__init__()
        self.dyn_embedding = nn.Sequential(
            nn.Linear(n_dynamic_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.month_embed = nn.Embedding(12, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(n_static_features, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_dynamic, x_static):
        B, T, _ = x_dynamic.shape
        x = self.dyn_embedding(x_dynamic)
        x = x + self.month_embed(torch.arange(T, device=x.device)).unsqueeze(0)
        x = self.transformer(x)
        h_dyn = x.mean(dim=1)
        h_sta = self.static_mlp(x_static)
        return self.head(torch.cat([h_dyn, h_sta], dim=-1)).squeeze(-1)
