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

        # Better initialization for a_raw (log-space spread)
        self.a_raw = nn.Parameter(torch.linspace(-3.0, -0.1, d_state))
        self.b = nn.Parameter(torch.randn(d_state, in_dim) * 0.01)
        self.c = nn.Parameter(torch.randn(out_dim, d_state) * 0.01)
        self.d = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)

        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor):
        """
        Vectorized training path using FFT-based convolution.
        u: (B, T, in_dim)
        return: (B, T, out_dim)
        """
        B, T, Din = u.shape
        assert Din == self.in_dim

        # A in discrete space: a = tanh(a_raw)
        # Note: Using tanh(a_raw) is a simple way to keep |a| < 1
        a = torch.tanh(self.a_raw)  # (d_state)

        # 1. Expand input (B, T, in_dim) -> (B, T, d_state)
        # B * u is (B, T, d_state)
        v = torch.einsum('bti,si->bts', u, self.b)

        # 2. Compute convolution kernels: k_t = a^t
        # K shape: (d_state, T)
        t = torch.arange(T, device=u.device, dtype=u.dtype)
        # Kernel k[s, t] = a[s]^t
        # a is (d_state), t is (T)
        k = torch.pow(a.unsqueeze(-1), t.unsqueeze(0))

        # 3. FFT Convolution (Linear convolution via padding to 2*T) 
        n_fft = 2**math.ceil(math.log2(2 * T - 1))
        V_f = torch.fft.rfft(v, n=n_fft, dim=1)
        K_f = torch.fft.rfft(k, n=n_fft, dim=1)
        
        # Multiply in frequency domain: (B, F, S) * (1, F, S)
        # K_f is (S, F), so we transpose or unsqueeze
        X_f = V_f * K_f.transpose(0, 1).unsqueeze(0)
        
        # Back to time domain
        x = torch.fft.irfft(X_f, n=n_fft, dim=1)[:, :T, :] # (B, T, d_state)

        # 4. Final output projection y = C x + D u
        y = torch.einsum('bts,os->bto', x, self.c) + torch.einsum('bti,oi->bto', u, self.d)
        
        return self.dropout(y)

    def step(self, u: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Stateful recurrence step for O(1) streaming inference.
        u: (B, 1, in_dim) or (B, in_dim)
        state: (B, d_state)
        """
        if u.dim() == 3:
            u = u.squeeze(1)
        if state is None:
            state = torch.zeros(u.size(0), self.d_state, device=u.device, dtype=u.dtype)
        
        a = torch.tanh(self.a_raw) # (S)
        # x_{t+1} = a*x_t + B*u_t
        new_state = state * a + torch.einsum('si,bi->bs', self.b, u)
        # y_t = C*x_t + D*u_t
        y = torch.einsum('os,bs->bo', self.c, new_state) + torch.einsum('oi,bi->bo', self.d, u)
        
        return y, new_state


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
            # Pre-Norm architecture
            z = ln(h)
            y = ssm(z)
            h = F.gelu(y) + h  # residual
        h = self.out_norm(h)
        return h  # (B, T, hidden)

    def step(self, x: torch.Tensor, states: Optional[list] = None):
        """
        Cascaded step for encoder.
        x: (B, hidden)
        states: List of tensors
        """
        if states is None:
            states = [None] * len(self.layers)
        
        new_states = []
        h = self.in_proj(x.unsqueeze(1)).squeeze(1) # (B, hidden)
        for i, (ssm, ln) in enumerate(zip(self.layers, self.norms)):
            z = ln(h)
            y, new_s = ssm.step(z, states[i])
            h = F.gelu(y) + h
            new_states.append(new_s)
        
        h = self.out_norm(h)
        return h, new_states


class ECGSSMClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, d_state: int = 64, hidden: int = 64, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = SSMEncoder(d_state=d_state, hidden=hidden, depth=depth, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T) or (B, T, 1)
        h = self.encoder(x)  # (B, T, H)
        # Robust Pooling: Concat Global Avg and Global Max
        avg_p = h.mean(dim=1)
        max_p = h.max(dim=1)[0]
        combined = torch.cat([avg_p, max_p], dim=-1) # (B, 2*H)
        logits = self.head(combined)
        return logits

    def step(self, x: torch.Tensor, states: Optional[list] = None):
        # x is a single sample (B, 1)
        h, new_states = self.encoder.step(x, states)
        # In pure streaming mode, we can't do global pooling over T
        # We just return the current hidden state projected to classes
        # This assumes the 'head' is trained or compatible. 
        # For simplicity, we'll project the single hidden state.
        # But wait, the pooling head is (2*H). 
        # A better way for streaming: return a rolling pool or just the raw projection.
        # Let's use a simpler head for the step or just return hidden for now.
        return h, new_states

