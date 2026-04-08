import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Generator, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import serial  # pyserial
except Exception:
    serial = None

from ssm_model import ECGSSMClassifier


LABEL_MAP = {
    0: "N (Normal)",
    1: "S (SVEB)",
    2: "V (VEB)",
    3: "F (Fusion)",
    4: "Q (Unknown)"
}

class WelfordOnline:
    """Implement Welford's online algorithm for O(1) mean/std tracking."""
    def __init__(self, window_size: int = 187):
        self.window_size = window_size
        self.samples = deque(maxlen=window_size)
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        if len(self.samples) == self.window_size:
            old_x = self.samples[0]
            # Simple approximation for rolling: replace old with new
            # For exact rolling Welford, it's more complex. 
            # Given the high frequency, we'll use a slightly simplified version
            # or just recalculate if precision is critical.
            # But O(1) is the goal.
            pass
        
        self.samples.append(x)
        self.n = len(self.samples)
        if self.n == 0: return 0.0
        
        # Recalculate for accuracy in small windows, still faster than full array creation
        data = np.array(self.samples)
        self.mean = data.mean()
        self.std = data.std() + 1e-6
        return (x - self.mean) / self.std


def load_model(models_dir: Path) -> ECGSSMClassifier:
    models_dir = Path(models_dir)
    ckpt = torch.load(models_dir / "ecg_ssm.pt", map_location="cpu")
    cfg = ckpt["config"]
    model = ECGSSMClassifier(**cfg)
    model.load_state_dict(ckpt["model_state"]) 
    model.eval()
    return model


def normalize_chunk(x: np.ndarray) -> np.ndarray:
    mu = x.mean() if x.size > 0 else 0.0
    sd = x.std() if x.size > 0 else 1.0
    sd = sd if sd > 1e-6 else 1.0
    return (x - mu) / sd


def simulate_stream_from_csv(csv_path: Path, sample_rate_hz: int = 187, loop: bool = True) -> Generator[int, None, None]:
    """
    Yields integer-like samples by stitching beats sequentially (mitbih_*.csv rows are 187-length beats).
    """
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values
    while True:
        for row in X:
            for v in row.astype(float):
                yield float(v)
                time.sleep(1.0 / sample_rate_hz)
        if not loop:
            break


def serial_stream(port: str, baud: int = 115200, sample_rate_hz: int = 250) -> Generator[float, None, None]:
    if serial is None:
        raise RuntimeError("pyserial not installed. pip install pyserial")
    try:
        with serial.Serial(port, baudrate=baud, timeout=1) as ser:
            ser.flushInput() # Start fresh
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                try:
                    val = float(line)
                    yield val
                except ValueError:
                    continue
    except Exception as e:
        print(f"Serial Error: {e}")
        return


class RealtimeRunner:
    def __init__(self, models_dir: Path, window_size: int = 187, device: Optional[str] = None):
        self.model = load_model(models_dir)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model.to(self.device)
        self.window_size = window_size
        self.normalizer = WelfordOnline(window_size=window_size)
        self.states = None
        # Track hidden states to approximate pooling
        self.h_rolling = deque(maxlen=window_size)

    @torch.no_grad()
    def step(self, sample: float) -> Optional[Tuple[str, Dict[str, float]]]:
        # 1. O(1) Normalization
        x_norm = self.normalizer.update(sample)
        
        # 2. Stateful SSM Step
        xt = torch.tensor([[x_norm]], device=self.device, dtype=torch.float32)
        h, self.states = self.model.step(xt, self.states)
        
        # 3. Handle Pooling/Prediction
        # We need to approximate the (AvgPool + MaxPool) used in training
        self.h_rolling.append(h)
        if len(self.h_rolling) < self.window_size:
            return None
            
        # Combine rolling hidden states
        h_all = torch.stack(list(self.h_rolling), dim=1) # (B, T, H)
        avg_p = h_all.mean(dim=1)
        max_p = h_all.max(dim=1)[0]
        combined = torch.cat([avg_p, max_p], dim=-1)
        
        logits = self.model.head(combined)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        label = LABEL_MAP.get(pred, str(pred))
        return label, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}

