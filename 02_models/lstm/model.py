# 02_models/lstm/model.py
"""Simple single-branch LSTM for glacier SMB prediction.

Architecture:
  LSTM(n_dynamic_features, hidden_dim, num_layers) → last hidden state
  concat with x_sta → Linear head → scalar SMB
"""
import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, n_dynamic_features: int, n_static_features: int,
                 hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_dynamic_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + n_static_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_dyn: torch.Tensor, x_sta: torch.Tensor) -> torch.Tensor:
        # x_dyn: (N, 12, n_dynamic_features)
        # x_sta: (N, n_static_features)
        _, (h_n, _) = self.lstm(x_dyn)
        h = h_n[-1]                                  # last layer: (N, hidden_dim)
        out = self.head(torch.cat([h, x_sta], dim=1))
        return out.squeeze(1)                        # (N,)
