import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSSMLayer(nn.Module):
    """
    A lightweight diagonal state-space layer.

    Discrete-time update per time-step t:
      x_{t+1} = a * x_t + b * u_t
      y_t     = c * x_t + d * u_t

    Where a, b, c, d are learnable (a is constrained to (-1,1) via tanh for stability).
    We run the recurrence over the time dimension and return y (sequence).
    """

    def __init__(self, d_state: int, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.d_state = d_state
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Parameters for diagonal SSM
        self.a_raw = nn.Parameter(torch.zeros(d_state))  # mapped with tanh
        self.b = nn.Parameter(torch.randn(d_state, in_dim) * 0.1)
        self.c = nn.Parameter(torch.randn(out_dim, d_state) * 0.1)
        self.d = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor):
        """
        u: (B, T, in_dim)
        return: (B, T, out_dim)
        """
        B, T, Din = u.shape
        assert Din == self.in_dim

        a = torch.tanh(self.a_raw)  # stability: |a|<1
        x = torch.zeros(B, self.d_state, device=u.device, dtype=u.dtype)

        ys = []
        for t in range(T):
            u_t = u[:, t, :]  # (B, in_dim)
            x = x * a + torch.einsum('sd,bd->bs', self.b, u_t)  # (B, d_state)
            y_t = torch.einsum('od,bd->bo', self.c, x) + torch.einsum('oi,bi->bo', self.d, u_t)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, out_dim)
        return self.dropout(y)


class SSMEncoder(nn.Module):
    """
    Stack a few SimpleSSMLayers with GELU and residuals.
    Accepts 1D sequences shaped (B, T) or (B, T, 1).
    """

    def __init__(self, d_state: int = 64, hidden: int = 64, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([
            SimpleSSMLayer(d_state=d_state, in_dim=hidden, out_dim=hidden, dropout=dropout)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.in_proj(x)
        for ssm, ln in zip(self.layers, self.norms):
            y = ssm(h)
            h = ln(F.gelu(y) + h)  # residual
        h = self.out_norm(h)
        return h  # (B, T, hidden)


class ECGSSMClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, d_state: int = 64, hidden: int = 64, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = SSMEncoder(d_state=d_state, hidden=hidden, depth=depth, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # pool over T
            nn.Flatten(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T) or (B, T, 1)
        h = self.encoder(x)  # (B, T, H)
        h = h.transpose(1, 2)  # (B, H, T)
        logits = self.head(h)
        return logits

