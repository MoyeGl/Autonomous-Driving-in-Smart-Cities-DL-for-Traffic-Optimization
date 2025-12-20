"""
model.py
Author: Moye Nyuysoni
Multi-task LSTM: predict next speed (regression) + lane-change activity (classification).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class LSTMTrafficMultiTask(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.speed_head = nn.Linear(hidden, 1)   # regression
        self.lc_head = nn.Linear(hidden, 1)      # logits

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        z = self.shared(h)
        speed = self.speed_head(z).squeeze(-1)
        lc_logits = self.lc_head(z).squeeze(-1)
        return speed, lc_logits
