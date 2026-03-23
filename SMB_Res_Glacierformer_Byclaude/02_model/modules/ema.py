"""
EMA1D — 高效多尺度时序注意力模块

原始论文: Tang et al., "EMA: Efficient Multi-Scale Attention for Efficient
           Network Design", ICASSP 2023
原始实现: https://github.com/AIFengheshu/Plug-play-modules
          文件: EMA.py

本版本将原始 2D 空间多尺度注意力适配为 1D 时序版本：
  - 使用不同卷积核尺寸的深度卷积捕捉不同时间窗口内的依赖关系
  - Squeeze-and-Excitation 风格的门控机制自动学习各时间尺度的重要程度
  - 残差结构确保梯度稳定

物理意义（冰川 SMB 预测）:
  Scale = 1 (月): 单月气候异常（热浪、暴雪等极端事件）
  Scale = 3 (季): 季节性积累/消融模式（冬季积雪 vs 夏季消融）
  Scale = 6 (半年): 年内积累期与消融期的整体对比

  不同冰川对这三个尺度的敏感程度不同（如海洋性冰川更受月际变化影响），
  SE 门控能自适应地强调每个样本最相关的时间尺度。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class EMA1D(nn.Module):
    """
    Efficient Multi-Scale Attention for 1D time series.

    Args:
        d_model : 特征维度
        scales  : 多尺度卷积核尺寸列表，默认 [1, 3, 6]（月、季、半年）
        factor  : SE 门控中间维度压缩比（d_model // factor）
        dropout : Dropout 比率
    """

    def __init__(self, d_model: int, scales: List[int] = None,
                 factor: int = 4, dropout: float = 0.1):
        super().__init__()
        if scales is None:
            scales = [1, 3, 6]
        self.scales  = scales
        self.d_model = d_model
        n_scales     = len(scales)

        # 每个时间尺度的深度可分离卷积
        # groups=d_model 实现深度卷积（通道间独立，参数少）
        self.scale_convs = nn.ModuleList()
        for s in scales:
            ks  = s if s > 1 else 1
            pad = ks // 2
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=ks,
                              padding=pad, groups=d_model, bias=False),
                    nn.Conv1d(d_model, d_model, kernel_size=1),  # 通道混合
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                )
            )

        # SE 门控：全局均值池化 → MLP → Sigmoid → 多尺度权重
        mid = max(d_model // factor, 16)
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model * n_scales, mid),
            nn.ReLU(),
            nn.Linear(mid, d_model * n_scales),
            nn.Sigmoid(),
        )

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model) 多尺度注意力加权特征
        """
        B, T, C  = x.shape
        residual = x
        x_t      = x.transpose(1, 2)   # (B, C, T) — Conv1d 格式

        # 每个时间尺度单独提取特征
        scale_outs = []
        for conv in self.scale_convs:
            feat = conv(x_t)                     # (B, C, T')
            # 确保输出长度与输入一致（奇数卷积核可能差 1）
            if feat.shape[-1] != T:
                feat = feat[..., :T]
            scale_outs.append(feat.transpose(1, 2))   # (B, T, C)

        # SE 门控：用全局均值上下文决定各尺度权重
        global_ctx = torch.cat(
            [f.mean(dim=1) for f in scale_outs], dim=-1   # (B, C*n_scales)
        )
        gates = self.scale_gate(global_ctx)               # (B, C*n_scales)
        gates = gates.view(B, len(self.scales), 1, C)     # (B, n_scales, 1, C)

        # 加权求和
        stacked = torch.stack(scale_outs, dim=1)          # (B, n_scales, T, C)
        out     = (stacked * gates).sum(dim=1)            # (B, T, C)

        # 输出投影 + 残差 + LayerNorm
        out = self.dropout(self.out_proj(out))
        return self.norm(out + residual)
