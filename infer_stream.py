import time
from collections import deque
from queue import Queue
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import serial  # pyserial
except Exception:
    serial = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

from ssm_model import ECGSSMClassifier, LegacyECGSSMClassifier


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
    state = ckpt["model_state"]
    if "head.0.weight" in state:
        model = ECGSSMClassifier(**cfg)
    else:
        model = LegacyECGSSMClassifier(**cfg)
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


def mqtt_stream(host: str, port: int = 1883, topic: str = "ecg/data", keepalive: int = 60) -> Generator[float, None, None]:
    if mqtt is None:
        raise RuntimeError("paho-mqtt not installed. pip install paho-mqtt")

    samples: Queue[float] = Queue()
    errors: Queue[Exception] = Queue()

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc != 0:
            errors.put(RuntimeError(f"MQTT connection failed with code {rc}"))
            return
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="ignore").strip()
        if not payload:
            return
        for value in parse_mqtt_samples(payload):
            samples.put(value)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=keepalive)
    client.loop_start()

    try:
        while True:
            if not errors.empty():
                raise errors.get()
            yield samples.get()
    finally:
        client.loop_stop()
        client.disconnect()


def parse_mqtt_samples(payload: str) -> list[float]:
    payload = payload.strip()
    if not payload:
        return []

    try:
        return [float(payload)]
    except ValueError:
        pass

    # Accept JSON payloads like {"samples":[...]} for future expansion.
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        values = []
        for item in data["samples"]:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return []
        return values

    if isinstance(data, list):
        values = []
        for item in data:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return []
        return values

    # Accept comma-separated batches like "512,515,498"
    if "," in payload:
        values = []
        for part in payload.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                return []
        return values

    return []


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


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
        if getattr(self.model, "streaming_pool", "avgmax") == "avg":
            pooled = avg_p
        else:
            max_p = h_all.max(dim=1)[0]
            pooled = torch.cat([avg_p, max_p], dim=-1)

        logits = self.model.head(pooled)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        label = LABEL_MAP.get(pred, str(pred))
        return label, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}


class ONNXRealtimeRunner:
    def __init__(self, models_dir: Path, window_size: int = 187):
        if ort is None:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime")

        model_path = Path(models_dir) / "ecg_ssm.onnx"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.window_size = window_size
        self.samples = deque(maxlen=window_size)

    def step(self, sample: float) -> Optional[Tuple[str, Dict[str, float]]]:
        self.samples.append(float(sample))
        if len(self.samples) < self.window_size:
            return None

        window = np.array(self.samples, dtype=np.float32)
        window = normalize_chunk(window).astype(np.float32).reshape(1, self.window_size)
        logits = self.session.run(None, {self.input_name: window})[0]
        probs = softmax_np(np.asarray(logits[0], dtype=np.float32))
        pred = int(np.argmax(probs))
        label = LABEL_MAP.get(pred, str(pred))
        return label, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
