"""
双分支 LSTM 网络模型定义

架构参考 MassBalanceMachine Two-Branch Network：
  时序分支 (LSTM)  : (B, 12, F_dynamic) → 最后时步隐状态 → (B, lstm_hidden)
  空间分支 (MLP)   : (B, F_static)       → 全连接层      → (B, static_hidden)
  融合层           : Concat → Dense → (B, 1) — 年度 SMB (mm w.e.)

依赖: torch
"""
import torch
import torch.nn as nn


class TwoBranchLSTM(nn.Module):
    """
    双分支网络：时序 LSTM + 静态 MLP，融合预测年度 SMB。

    参数
    ----
    n_dynamic_features  : 动态特征数（月度气候变量数，F_dynamic=15）
    n_static_features   : 静态特征数（地形特征数，F_static=5）
    lstm_hidden_dim     : LSTM 隐藏状态维度
    lstm_num_layers     : LSTM 堆叠层数
    bidirectional       : 是否使用双向 LSTM
    static_mlp_hidden   : 静态 MLP 隐藏层维度
    static_mlp_layers   : 静态 MLP 深度（含输入层后的隐藏层数）
    dropout             : Dropout 比率（应用于 LSTM 层间和 MLP）
    """

    def __init__(
        self,
        n_dynamic_features: int,
        n_static_features: int,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        bidirectional: bool = False,
        static_mlp_hidden: int = 64,
        static_mlp_layers: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        # ── 时序分支：LSTM ────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = n_dynamic_features,
            hidden_size  = lstm_hidden_dim,
            num_layers   = lstm_num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = dropout if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(p=dropout)

        # ── 空间分支：MLP（全连接）───────────────────────────────────────────
        mlp_layers = []
        in_dim = n_static_features
        for _ in range(static_mlp_layers):
            mlp_layers += [
                nn.Linear(in_dim, static_mlp_hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = static_mlp_hidden
        self.static_mlp = nn.Sequential(*mlp_layers)

        # ── 融合层：Concat → Dense → 输出 ────────────────────────────────────
        fusion_in_dim = lstm_out_dim + static_mlp_hidden
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_dynamic: torch.Tensor,   # (B, 12, F_dynamic)
        x_static:  torch.Tensor,   # (B, F_static)
    ) -> torch.Tensor:             # (B,)
        """前向传播，返回 (B,) 形状的 SMB 预测值（mm w.e.）。"""
        # 时序分支
        lstm_out, _ = self.lstm(x_dynamic)          # (B, 12, lstm_out_dim)
        lstm_feat   = lstm_out[:, -1, :]             # 取最后时步 (B, lstm_out_dim)
        lstm_feat   = self.lstm_dropout(lstm_feat)

        # 空间分支
        static_feat = self.static_mlp(x_static)     # (B, static_mlp_hidden)

        # 融合预测
        combined = torch.cat([lstm_feat, static_feat], dim=-1)   # (B, fusion_in)
        out      = self.fusion_head(combined)                     # (B, 1)
        return out.squeeze(-1)                                    # (B,)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, F_dyn, F_sta = 8, 12, 15, 5
    model = TwoBranchLSTM(n_dynamic_features=F_dyn, n_static_features=F_sta)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n可训练参数: {total_params:,}")

    x_dyn = torch.randn(B, T, F_dyn)
    x_sta = torch.randn(B, F_sta)
    y_hat = model(x_dyn, x_sta)
    print(f"输出形状: {y_hat.shape}  (预期: ({B},))")
