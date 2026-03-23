"""
WTConv1D — 1D 小波变换卷积模块

原始论文: Finder et al., "WTConv: Broadening the Field of View of Convolutional
           Networks through Wavelet Transformation", ECCV 2024
原始实现: https://github.com/BGU-CS-VIL/WTConv （2D 图像版）

本版本将 2D 小波卷积适配为 1D 时序版本，使用 Haar 小波对 12 个月的
气候序列进行多尺度分解，分离低频（年际趋势）与高频（月际变异）信息。

物理意义:
  低频分量 → 年际气候趋势（厄尔尼诺、PDO等多年信号）
  高频分量 → 月际极端气候事件（异常热浪、强降雪）
  两者对冰川物质平衡均有重要影响，分开处理能更好地捕捉不同尺度的驱动机制。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WTConv1D(nn.Module):
    """
    1D Haar Wavelet Convolution for temporal sequences.

    对输入序列 (B, T, C) 进行多级 Haar 小波分解：
      1. 低通 (low-pass) : 取相邻时步均值 → 捕捉平滑趋势
      2. 高通 (high-pass): 取相邻时步差值 → 捕捉局部变化
    分别通过点卷积处理后上采样至原始长度，合并输出。

    Args:
        channels  : 特征维度 C（输入输出维度相同，残差设计）
        seq_len   : 时序长度 T（默认 12 个月）
        wt_levels : 小波分解层数（默认 2，12月→6→3）
        dropout   : Dropout 比率
    """

    def __init__(self, channels: int, seq_len: int = 12,
                 wt_levels: int = 2, dropout: float = 0.1):
        super().__init__()
        # 实际分解层数不超过 log2(seq_len)
        max_levels = 1
        t = seq_len
        while t > 2:
            t //= 2
            max_levels += 1
        self.wt_levels = min(wt_levels, max_levels - 1)
        self.channels  = channels

        # 点卷积：处理低频和高频分量（共享参数 → 减少参数量）
        self.low_conv  = nn.Conv1d(channels, channels, kernel_size=1)
        self.high_conv = nn.Conv1d(channels, channels, kernel_size=1)

        # 合并低频 + 高频 → 输出
        self.merge = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _haar_decompose(x: torch.Tensor):
        """
        对 x: (B, C, T) 执行一级 Haar 小波分解。
        Returns:
            low : (B, C, T//2) 低频分量（趋势）
            high: (B, C, T//2) 高频分量（细节）
        """
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1), mode='replicate')
        low  = (x[..., 0::2] + x[..., 1::2]) * (2 ** -0.5)
        high = (x[..., 0::2] - x[..., 1::2]) * (2 ** -0.5)
        return low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C) 小波增强后的特征
        """
        residual = x
        x_t = x.transpose(1, 2)   # (B, C, T) — Conv1d 期望 channel-first
        T   = x_t.shape[-1]

        # 多级小波分解，收集各级高频分量
        low = x_t
        high_list = []
        for _ in range(self.wt_levels):
            low, high = self._haar_decompose(low)
            high_list.append(high)

        # 处理低频：点卷积 + 上采样至原始长度
        low_feat = self.low_conv(low)                                    # (B, C, T/2^k)
        low_feat = F.interpolate(low_feat, size=T,
                                 mode='linear', align_corners=False)     # (B, C, T)

        # 处理高频：逐级点卷积 + 上采样，累加
        high_feat = torch.zeros_like(low_feat)
        for h in high_list:
            h_proc = self.high_conv(h)
            h_proc = F.interpolate(h_proc, size=T,
                                   mode='linear', align_corners=False)
            high_feat = high_feat + h_proc

        # 合并低频 + 高频特征
        merged = self.merge(torch.cat([low_feat, high_feat], dim=1))     # (B, C, T)
        out    = self.dropout(merged.transpose(1, 2)) + residual          # (B, T, C)
        return out
