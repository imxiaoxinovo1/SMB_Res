"""
V1: VanillaTransformer — 双分支标准 Transformer（最简基线）

架构:
  时序分支: Linear(15→d) + 可学习位置编码 → TransformerEncoder(2层,4头) → MeanPool → (B,d)
  静态分支: MLP(10→d//2→d, GELU) → (B,d)
  融合:     Concat(B,2d) → Linear(2d→d) → GELU → Dropout → Linear(d→1)

设计原则:
  - 全部 GELU（无 ReLU，避免负激活截断引起偏差）
  - 全部 LayerNorm（无 BatchNorm，对小 batch 稳定）
  - 无任何额外模块（作为后续消融实验的干净基线）
"""
import torch
import torch.nn as nn


class VanillaTransformer(nn.Module):
    MODEL_NAME = "v1_vanilla"

    def __init__(
        self,
        n_dynamic_features: int = 15,
        n_static_features:  int = 10,
        d_model:            int = 64,
        n_heads:            int = 4,
        n_encoder_layers:   int = 2,
        ff_dim:             int = 256,
        dropout:            float = 0.15,
        **kwargs,   # 忽略其他模型特有参数（训练脚本统一接口）
    ):
        super().__init__()

        # ── 时序分支 ──────────────────────────────────────────────────────────
        # 1. 输入嵌入
        self.dyn_embedding = nn.Sequential(
            nn.Linear(n_dynamic_features, d_model),
            nn.LayerNorm(d_model),
        )
        # 2. 可学习月份位置编码
        self.month_embed = nn.Embedding(12, d_model)

        # 3. 标准 Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = n_heads,
            dim_feedforward = ff_dim,
            dropout     = dropout,
            activation  = 'gelu',
            batch_first = True,
            norm_first  = True,   # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)

        # ── 静态分支 ──────────────────────────────────────────────────────────
        self.static_mlp = nn.Sequential(
            nn.Linear(n_static_features, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 融合 + 预测头 ─────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_dynamic: torch.Tensor,
                x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_dynamic: (B, 12, 15)
            x_static:  (B, 10)
        Returns:
            (B,) SMB 预测值 mm w.e.
        """
        B, T, _ = x_dynamic.shape

        # 时序分支
        x = self.dyn_embedding(x_dynamic)                          # (B,12,d)
        months = torch.arange(T, device=x.device)
        x = x + self.month_embed(months).unsqueeze(0)              # (B,12,d)
        x = self.transformer(x)                                    # (B,12,d)
        h_dyn = x.mean(dim=1)                                      # (B,d)

        # 静态分支
        h_sta = self.static_mlp(x_static)                         # (B,d)

        # 融合 + 预测
        out = self.head(torch.cat([h_dyn, h_sta], dim=-1))        # (B,1)
        return out.squeeze(-1)                                     # (B,)


if __name__ == '__main__':
    model = VanillaTransformer()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VanillaTransformer 参数量: {total:,}")
    x_dyn = torch.randn(8, 12, 15)
    x_sta = torch.randn(8, 10)
    print(f"输出 shape: {model(x_dyn, x_sta).shape}")
