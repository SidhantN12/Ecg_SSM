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
    with serial.Serial(port, baudrate=baud, timeout=1) as ser:
        while True:
            line = ser.readline().strip()
            if not line:
                continue
            try:
                val = float(line)
                yield val
                # Optionally sleep if device streams faster than sample_rate_hz
            except ValueError:
                continue


class RealtimeRunner:
    def __init__(self, models_dir: Path, window_size: int = 187, device: Optional[str] = None):
        self.model = load_model(models_dir)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model.to(self.device)
        self.window: Deque[float] = deque(maxlen=window_size)
        self.window_size = window_size

    @torch.no_grad()
    def step(self, sample: float) -> Optional[Tuple[str, Dict[str, float]]]:
        self.window.append(sample)
        if len(self.window) < self.window_size:
            return None
        x = np.array(self.window, dtype=np.float32)
        x = normalize_chunk(x)
        xb = torch.from_numpy(x).unsqueeze(0)  # (1, T)
        xb = xb.to(self.device)
        logits = self.model(xb)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        label = LABEL_MAP.get(pred, str(pred))
        return label, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}

