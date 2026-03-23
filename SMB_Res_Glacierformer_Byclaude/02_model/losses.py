"""
GlacioFormer 损失函数

L_total = L_MSE + α × L_physics

L_physics: 温度-SMB 单调性软约束
  物理先验: 年均气温升高 → 消融增加 → SMB 应降低（更负）
  实现: 在 batch 内计算温度与预测 SMB 的 Pearson 相关系数，
        若相关系数为正（违反物理先验），则给予惩罚。
  注意: 归一化后的温度特征 t̃ = (t - μ) / σ，方向不变，
        因此相关系数的符号判断仍然有效。
"""
import torch
import torch.nn as nn
from typing import Optional, List


class PhysicsInformedLoss(nn.Module):
    """
    MSE + Temperature-SMB Monotonicity Constraint.

    Args:
        alpha          : 物理约束项权重（默认 0.1）
        temp_var_name  : 用于约束的温度变量名（在 climate_cols 中查找）
        tolerance      : 允许的相关系数正值容差（默认 0.1，
                         避免因 batch 采样随机性导致不稳定惩罚）
    """

    def __init__(
        self,
        alpha:         float = 0.1,
        temp_var_name: str   = 'temperature_2m',
        tolerance:     float = 0.1,
    ):
        super().__init__()
        self.alpha         = alpha
        self.temp_var_name = temp_var_name
        self.tolerance     = tolerance
        self.mse           = nn.MSELoss()

    def forward(
        self,
        y_pred:      torch.Tensor,           # (B,) 预测值
        y_true:      torch.Tensor,           # (B,) 真实值
        x_dynamic:   torch.Tensor,           # (B, 12, F_dyn) 归一化动态特征
        climate_cols: Optional[List[str]] = None,  # 动态特征列名
    ) -> dict:
        """
        计算总损失和各分量。

        Returns:
            dict with keys: 'total', 'mse', 'physics'
        """
        # ── MSE 主损失 ────────────────────────────────────────────────────────
        loss_mse = self.mse(y_pred, y_true)

        # ── 物理约束：温度-SMB 单调性 ─────────────────────────────────────────
        loss_physics = self._temperature_monotonicity(
            y_pred, x_dynamic, climate_cols
        )

        loss_total = loss_mse + self.alpha * loss_physics

        return {
            'total':   loss_total,
            'mse':     loss_mse.detach(),
            'physics': loss_physics.detach(),
        }

    def _temperature_monotonicity(
        self,
        y_pred:      torch.Tensor,
        x_dynamic:   torch.Tensor,
        climate_cols: Optional[List[str]],
    ) -> torch.Tensor:
        """
        温度-SMB Pearson 相关系数约束。

        归一化温度特征 → 年均值 → 与 y_pred 计算相关系数
        → 若相关系数 > tolerance 则给予惩罚（relu）。

        Returns:
            physics_loss: scalar tensor
        """
        # 确定温度特征的列索引
        temp_idx = 0   # 默认使用第 0 列（temperature_2m 是 MONTHLY_CLIMATE_VARS[0]）
        if climate_cols is not None:
            try:
                temp_idx = list(climate_cols).index(self.temp_var_name)
            except ValueError:
                pass  # 找不到则使用默认索引

        # 年均温度（归一化后）: (B,)
        mean_temp = x_dynamic[:, :, temp_idx].mean(dim=1)

        # Pearson 相关系数（在 batch 内计算）
        t = mean_temp - mean_temp.mean()
        s = y_pred - y_pred.mean()

        t_std = t.std() + 1e-8
        s_std = s.std() + 1e-8

        corr = (t * s).mean() / (t_std * s_std)   # [-1, 1]

        # 惩罚：温度与 SMB 正相关（物理上不合理）
        # relu(corr - (-tolerance)) = relu(corr + tolerance)
        # → 仅当 corr > -tolerance（即接近或大于0）时才惩罚
        return torch.relu(corr + self.tolerance)
