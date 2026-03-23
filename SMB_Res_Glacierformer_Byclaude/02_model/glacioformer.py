"""
GlacioFormer — 冰川物质平衡预测双分支 Transformer 网络

架构概览:
  时序分支 (Temporal Branch):
    Input Embedding (15→64) → Month Positional Encoding
    → WTConv1D (小波多尺度分解)
    → FFTTransformerEncoderBlock × 2 (频域增强 Transformer)
    → EMA1D (多尺度时序注意力: 月/季/半年)
    → Global Average Pooling → (B, 64)

  静态分支 (Static Branch):
    MLP (5→32→64) → SimAM1D (无参数特征注意力) → (B, 64)

  融合层 (Fusion):
    CrossModalFreqFusion → (B, 64)

  预测头 (Head):
    Linear(64→32) → GELU → Dropout → Linear(32→1) → (B,)

参考模块来源:
  - WTConv1D      : Finder et al., ECCV 2024 (adapted to 1D)
  - FFTTransformer: arXiv 2025 (adapted to 1D time series)
  - EMA1D         : Tang et al., ICASSP 2023 (adapted to 1D)
  - SimAM1D       : Yang et al., ICML 2021 (adapted to 1D)
  - FreqFusion    : Chen et al., TPAMI 2024 (adapted to cross-modal)
  Plug-play-modules: https://github.com/AIFengheshu/Plug-play-modules
"""
import torch
import torch.nn as nn
from typing import List

from modules.simam import SimAM1D
from modules.wtconv import WTConv1D
from modules.fft_transformer import FFTTransformerEncoderBlock
from modules.ema import EMA1D
from modules.freq_fusion import CrossModalFreqFusion


class GlacioFormer(nn.Module):
    """
    GlacioFormer: Frequency-Aware Dual-Branch Transformer for Glacier SMB.

    Args:
        n_dynamic_features : 动态特征数（月度气候变量，默认 15）
        n_static_features  : 静态特征数（地形特征，默认 5）
        d_model            : Transformer 维度（默认 64）
        n_heads            : 注意力头数（默认 4，需整除 d_model）
        n_encoder_layers   : FFT-Transformer 堆叠层数（默认 2）
        ff_dim             : FFN 前馈维度（默认 256 = 4×d_model）
        ema_scales         : EMA 多尺度卷积核（默认 [1,3,6]）
        wt_levels          : 小波分解层数（默认 2）
        dropout            : Dropout 比率（默认 0.1）
    """

    def __init__(
        self,
        n_dynamic_features: int = 15,
        n_static_features:  int = 5,
        d_model:            int = 64,
        n_heads:            int = 4,
        n_encoder_layers:   int = 2,
        ff_dim:             int = 256,
        ema_scales:         List[int] = None,
        wt_levels:          int = 2,
        dropout:            float = 0.10,
    ):
        super().__init__()
        if ema_scales is None:
            ema_scales = [1, 3, 6]

        # ─────────────────────────────────────────────────────────────────────
        # 时序分支
        # ─────────────────────────────────────────────────────────────────────

        # 1. 输入嵌入：15 气候变量 → d_model
        self.dyn_embedding = nn.Sequential(
            nn.Linear(n_dynamic_features, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. 月份位置编码（可学习 Embedding，12 个月周期）
        self.month_embed = nn.Embedding(12, d_model)

        # 3. WTConv1D：小波多尺度特征提取
        self.wtconv = WTConv1D(
            channels=d_model, seq_len=12,
            wt_levels=wt_levels, dropout=dropout
        )

        # 4. FFT-Transformer：频域增强的 Transformer 编码器
        self.fft_transformer = nn.ModuleList([
            FFTTransformerEncoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_encoder_layers)
        ])

        # 5. EMA1D：多尺度时序注意力（月/季/半年）
        self.ema = EMA1D(d_model, scales=ema_scales, dropout=dropout)

        # 6. 全局均值池化 + 投影（(B,T,d) → (B,d)）
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # ─────────────────────────────────────────────────────────────────────
        # 静态分支
        # ─────────────────────────────────────────────────────────────────────

        # 7. MLP：地形特征升维
        self.static_mlp = nn.Sequential(
            nn.Linear(n_static_features, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 8. SimAM1D：无参数特征注意力
        self.simam = SimAM1D()

        # ─────────────────────────────────────────────────────────────────────
        # 融合层
        # ─────────────────────────────────────────────────────────────────────

        # 9. CrossModalFreqFusion：跨模态频率感知融合
        self.freq_fusion = CrossModalFreqFusion(
            d_dyn=d_model, d_sta=d_model, d_out=d_model, dropout=dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # 预测头
        # ─────────────────────────────────────────────────────────────────────

        # 10. 回归头
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x_dynamic: torch.Tensor,   # (B, 12, 15)
        x_static:  torch.Tensor,   # (B, 5)
    ) -> torch.Tensor:             # (B,)
        """前向传播，返回 (B,) 的年度 SMB 预测值（mm w.e.）。"""
        B, T, _ = x_dynamic.shape

        # ──────────────────────── 时序分支 ────────────────────────────────────
        # 输入嵌入
        x = self.dyn_embedding(x_dynamic)           # (B, 12, d_model)

        # 月份位置编码
        months = torch.arange(T, device=x.device)  # (12,) → [0,1,...,11]
        x = x + self.month_embed(months).unsqueeze(0)  # (B, 12, d_model)

        # 小波多尺度分解
        x = self.wtconv(x)                          # (B, 12, d_model)

        # FFT-Transformer 编码
        for block in self.fft_transformer:
            x = block(x)                            # (B, 12, d_model)

        # 多尺度时序注意力
        x = self.ema(x)                             # (B, 12, d_model)

        # 全局均值池化 → 时序上下文向量
        h_dyn = x.mean(dim=1)                       # (B, d_model)
        h_dyn = self.temporal_pool(h_dyn)           # (B, d_model)

        # ──────────────────────── 静态分支 ────────────────────────────────────
        h_sta = self.static_mlp(x_static)           # (B, d_model)
        h_sta = self.simam(h_sta)                   # (B, d_model)

        # ──────────────────────── 跨模态融合 ──────────────────────────────────
        h_fused = self.freq_fusion(h_dyn, h_sta)    # (B, d_model)

        # ──────────────────────── 预测输出 ────────────────────────────────────
        out = self.head(h_fused)                    # (B, 1)
        return out.squeeze(-1)                      # (B,)


# ── 快速结构验证 ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import GLACIOFORMER_PARAMS, N_CLIMATE_FEATURES, N_STATIC_FEATURES

    p = GLACIOFORMER_PARAMS
    model = GlacioFormer(
        n_dynamic_features = N_CLIMATE_FEATURES,
        n_static_features  = N_STATIC_FEATURES,
        d_model            = p['d_model'],
        n_heads            = p['n_heads'],
        n_encoder_layers   = p['n_encoder_layers'],
        ff_dim             = p['ff_dim'],
        ema_scales         = p['ema_scales'],
        wt_levels          = p['wt_levels'],
        dropout            = p['dropout'],
    )

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GlacioFormer 可训练参数: {total:,}")

    # 前向传播测试
    B = 8
    x_dyn = torch.randn(B, 12, N_CLIMATE_FEATURES)
    x_sta = torch.randn(B, N_STATIC_FEATURES)
    y_hat = model(x_dyn, x_sta)
    print(f"输出 shape: {y_hat.shape}  (预期: ({B},))")
