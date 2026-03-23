"""
SimAM — 无参数通道注意力模块（适配 1D 特征向量）

原始论文: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
           for Convolutional Neural Networks", ICML 2021
原始实现: https://github.com/ZjjConan/SimAM

原版设计用于图像 (B, C, H, W) 的空间注意力。
本版本适配为对 1D 特征向量 (B, C) 的通道注意力，
用于静态地形分支的特征重加权。

物理意义: 在 5 个静态地形特征中，根据统计显著性自动强调
          偏离均值较远的特征（对当前 batch 而言更"异常"、更有区分度的特征）。
"""
import torch
import torch.nn as nn


class SimAM1D(nn.Module):
    """
    SimAM adapted for 1D feature vectors (B, C).

    零参数：不引入任何可学习权重，
    仅根据批内统计量动态计算每个特征维度的注意力权重。
    对小样本数据（如本项目的 745 条训练数据）友好，不会增加过拟合风险。

    Args:
        e_lambda: 正则化项，防止分母为零（默认 1e-4）
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C) 静态特征向量

        Returns:
            out: (B, C) 注意力加权后的特征，shape 与输入相同
        """
        # 批内均值 (B, 1)
        mu = x.mean(dim=1, keepdim=True)

        # 每个特征相对均值的偏差平方 (B, C)
        d2 = (x - mu).pow(2)

        # 批内方差估计 (B, 1)
        var = d2.mean(dim=1, keepdim=True)

        # 能量函数 e*：偏差越大 → 注意力越高
        # 公式: e* = d² / (4*(σ² + λ)) + 0.5
        e_star = d2 / (4.0 * (var + self.e_lambda)) + 0.5

        # Sigmoid 归一化到 (0.5, 1)，保证最低也有 50% 的权重
        return x * self.act(e_star)
