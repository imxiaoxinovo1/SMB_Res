"""
CrossModalFreqFusion — 跨模态频率感知融合模块

原始论文: Chen et al., "FreqFusion: Frequency-Aware Feature Fusion for Dense
           Image Prediction", TPAMI 2024
原始实现: https://github.com/AIFengheshu/Plug-play-modules
          文件: FreqFusion.py

本版本将原始"高分辨率 × 低分辨率空间特征融合"适配为
"动态气候上下文 × 静态地形上下文的跨模态融合"：

  原始: 空间高分辨率特征 + 空间低分辨率特征 → 融合特征
  本版: 时序气候上下文 (B, d_dyn) + 地形静态特征 (B, d_sta) → 融合特征 (B, d_out)

两路融合机制:
  ① 频率过滤分支: 地形特征调制气候特征的频域分量
     （不同海拔/面积的冰川对气候信号的低频/高频响应不同）
  ② 交叉注意力分支: 地形特征作为查询，检索气候上下文中最相关的信息

物理意义:
  高海拔冰川主要受年际低频气候变化驱动（长期增温趋势）；
  低海拔海洋性冰川更受月际高频变化影响（短期降水/温度波动）。
  FreqFusion 让模型通过地形特征自适应地决定关注哪些频率的气候信号。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFreqFusion(nn.Module):
    """
    Cross-Modal Frequency-Aware Fusion.

    两路并行融合后求和：
      Path 1 (频率过滤): FFT(h_dyn) × 静态调制 → iFFT → 频域过滤的气候特征
      Path 2 (交叉注意力): 地形特征 Q × 气候特征 K/V → 上下文感知的气候特征

    Args:
        d_dyn  : 动态（气候）分支输入维度
        d_sta  : 静态（地形）分支输入维度
        d_out  : 融合输出维度
        dropout: Dropout 比率
    """

    def __init__(self, d_dyn: int, d_sta: int,
                 d_out: int, dropout: float = 0.1):
        super().__init__()
        # ── 投影至统一维度 ────────────────────────────────────────────────────
        self.proj_dyn = nn.Linear(d_dyn, d_out)
        self.proj_sta = nn.Linear(d_sta, d_out)

        # ── Path 1: 频率过滤分支 ──────────────────────────────────────────────
        # 地形特征生成频率调制权重（有界在 (-1, 1)，避免过度放缩）
        self.freq_modulator = nn.Sequential(
            nn.Linear(d_sta, d_out),
            nn.Tanh(),
        )
        # 全局频率滤波器（可学习，初始化为 1 即不过滤）
        n_freqs = d_out // 2 + 1
        self.global_filter = nn.Parameter(torch.ones(n_freqs))

        # ── Path 2: 交叉注意力分支 ────────────────────────────────────────────
        # Q 来自地形（"我想了解什么"），K/V 来自气候（"我有什么信息"）
        self.cross_q = nn.Linear(d_sta, d_out)
        self.cross_k = nn.Linear(d_dyn, d_out)
        self.cross_v = nn.Linear(d_dyn, d_out)
        self.attn_scale = d_out ** -0.5

        # ── 输出融合 ──────────────────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h_dyn: torch.Tensor,
                h_sta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_dyn: (B, d_dyn) 时序气候上下文（由 GlobalAvgPool 后的 Transformer 输出）
            h_sta: (B, d_sta) 静态地形特征（由 SimAM 增强后的 MLP 输出）
        Returns:
            out: (B, d_out) 融合特征
        """
        # 投影至统一维度
        d = self.proj_dyn(h_dyn)   # (B, d_out)
        s = self.proj_sta(h_sta)   # (B, d_out)

        # ── Path 1: 地形调制频率过滤 ──────────────────────────────────────────
        # 对气候特征向量做 FFT（沿特征维度，感知各特征分量的频率结构）
        d_freq = torch.fft.rfft(d, dim=-1)                    # (B, n_freqs) complex

        # 地形特征生成调制因子 → 与全局滤波器相乘
        mod     = self.freq_modulator(h_sta)                  # (B, d_out)
        mod_freq = torch.fft.rfft(mod, dim=-1)                # (B, n_freqs) complex
        mod_amp  = torch.abs(mod_freq).clamp(0.1, 3.0)        # 振幅调制，限制范围

        gf = torch.sigmoid(self.global_filter).unsqueeze(0)   # (1, n_freqs)
        d_filtered = torch.fft.irfft(
            d_freq * gf * mod_amp, n=d.shape[-1]
        )                                                       # (B, d_out)

        # ── Path 2: 地形查询气候特征 ──────────────────────────────────────────
        Q = self.cross_q(h_sta).unsqueeze(1)   # (B, 1, d_out)
        K = self.cross_k(h_dyn).unsqueeze(1)   # (B, 1, d_out)
        V = self.cross_v(h_dyn).unsqueeze(1)   # (B, 1, d_out)

        attn     = torch.softmax(
            (Q @ K.transpose(-2, -1)) * self.attn_scale, dim=-1
        )                                       # (B, 1, 1)
        d_cross  = (attn @ V).squeeze(1)        # (B, d_out)

        # ── 拼接两路 + 输出投影 ───────────────────────────────────────────────
        fused = self.out_proj(
            torch.cat([d_filtered, d_cross], dim=-1)   # (B, d_out*2)
        )                                               # (B, d_out)

        # 残差连接（以地形特征为基准，保留地形信息）
        return self.norm(fused + s)
