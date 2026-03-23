"""
FFTTransformerEncoderBlock — 频域增强的 Transformer 编码器块

原始论文: arXiv 2025 "FFT-Transformer: Frequency-Enhanced Transformer
           for Time Series Forecasting"
原始实现: https://github.com/AIFengheshu/Plug-play-modules
          文件: FFTTransformerEncoderBlock.py

本版本将原始 2D 图像 Transformer 适配为 1D 时序版本：
  - 沿时间维度执行 1D 实值 FFT (rfft)
  - 振幅谱特征作为额外查询/键增强，帮助注意力机制感知周期性结构
  - 保留标准 Multi-Head Self-Attention + FFN 子层（Pre-LN 架构）

物理意义:
  气候序列在频域中呈现清晰的年周期峰值（基频 = 1/12 month⁻¹）。
  FFT 增强使注意力头能直接感知哪些月份携带主频能量，
  从而更精准地识别"对 SMB 贡献最大的季节"。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTTransformerEncoderBlock(nn.Module):
    """
    FFT-enhanced Transformer Encoder Block for 1D time series.

    架构:
        x → FFT Enhancement → Self-Attention (Pre-LN) → FFN (Pre-LN) → out

    FFT Enhancement:
        对输入沿时间轴做 rfft，取振幅谱插值至原始长度，
        经可学习投影后通过输入门控与原始特征融合。

    Args:
        d_model : 模型维度（需能被 n_heads 整除）
        n_heads : 注意力头数
        ff_dim  : FFN 前馈维度（通常 4 × d_model）
        dropout : Dropout 比率
    """

    def __init__(self, d_model: int, n_heads: int,
                 ff_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"

        # ── FFT 增强分支 ────────────────────────────────────────────────────
        self.freq_proj = nn.Linear(d_model, d_model)   # 振幅谱投影
        self.freq_gate = nn.Sequential(                 # 输入门控（避免 FFT 特征干扰）
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # ── Self-Attention 子层（Pre-LN）────────────────────────────────────
        self.norm1     = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop1     = nn.Dropout(dropout)

        # ── FFN 子层（Pre-LN）───────────────────────────────────────────────
        self.norm2     = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop2     = nn.Dropout(dropout)

        # ── 输出归一化 ───────────────────────────────────────────────────────
        self.norm_out  = nn.LayerNorm(d_model)

    def _fft_enhance(self, x: torch.Tensor) -> torch.Tensor:
        """
        用频域振幅谱增强时序特征。

        Args:
            x: (B, T, d_model)
        Returns:
            x_enhanced: (B, T, d_model) 频域增强后的特征
        """
        B, T, d = x.shape

        # rfft 沿时间轴，取振幅谱
        x_freq = torch.fft.rfft(x, dim=1)          # (B, T//2+1, d) complex
        x_amp  = torch.abs(x_freq)                  # (B, T//2+1, d) real

        # 插值至原始时间长度
        x_amp_up = F.interpolate(
            x_amp.transpose(1, 2),                  # (B, d, T//2+1)
            size=T, mode='linear', align_corners=False
        ).transpose(1, 2)                            # (B, T, d)

        # 可学习投影 + 输入门控融合
        freq_feat  = self.freq_proj(x_amp_up)       # (B, T, d)
        gate       = self.freq_gate(x)              # (B, T, d), 取值 (0, 1)
        return x + freq_feat * gate                 # 残差融合

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        # 1. FFT 频域增强
        x = self._fft_enhance(x)

        # 2. Self-Attention 子层（Pre-LN + 残差）
        x_norm      = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x           = x + self.drop1(attn_out)

        # 3. FFN 子层（Pre-LN + 残差）
        x = x + self.drop2(self.ffn(self.norm2(x)))

        return self.norm_out(x)
