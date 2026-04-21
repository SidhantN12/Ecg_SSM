import time
from collections import deque
from queue import Queue
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple
import json
import struct
import math

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

def crc8_python(data: bytes) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc

class WelfordOnline:
    """Implement a true O(1) sliding window Welford's algorithm for normalization."""
    def __init__(self, window_size: int = 187):
        self.window_size = window_size
        self.samples = deque(maxlen=window_size)
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> float:
        if len(self.samples) < self.window_size:
            # Filling the window
            self.samples.append(x)
            old_mean = self.mean
            self.mean += (x - old_mean) / len(self.samples)
            self.m2 += (x - old_mean) * (x - self.mean)
        else:
            # Sliding the window
            x_old = self.samples.popleft()
            self.samples.append(x)
            
            delta = x - x_old
            old_mean = self.mean
            self.mean += delta / self.window_size
            self.m2 += delta * (x - self.mean + x_old - old_mean)

        std = math.sqrt(self.m2 / len(self.samples)) if self.m2 > 0 else 1.0
        std = std if std > 1e-6 else 1.0
        return (x - self.mean) / std


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


def simulate_stream_from_csv(csv_path: Path, sample_rate_hz: int = 187, loop: bool = True) -> Generator[float, None, None]:
    """
    Yields samples by stitching beats sequentially from MIT-BIH CSV rows.
    Keep for testing/debugging purposes.
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
        payload = msg.payload
        if not payload or len(payload) < 7: # Header(1) + Seq(4) + Count(1) + CRC(1)
            return
        
        # Binary protocol decoding
        if payload[0] == 0xA5:
            header_size = 6 # Header + Seq + Count
            # 0xA5 (B) + Seq (I) + Count (B)
            magic, seq, count = struct.unpack("<BIB", payload[:6])
            
            # The data length is count * 4 (floats)
            data_end = header_size + count * 4
            if len(payload) < data_end + 1:
                return
            
            # Verify CRC
            received_crc = payload[data_end]
            calculated_crc = crc8_python(payload[:data_end])
            if received_crc != calculated_crc:
                print(f"CRC Mismatch: {received_crc} != {calculated_crc}")
                return
            
            # Unpack floats
            samples_data = payload[header_size:data_end]
            float_samples = struct.unpack(f"<{count}f", samples_data)
            for s in float_samples:
                samples.put(s)
        else:
            # Fallback for old/debug string messages (if any, though broken compatibility is intended)
            try:
                line = payload.decode("utf-8", errors="ignore").strip()
                for val in parse_mqtt_samples(line):
                    samples.put(val)
            except Exception:
                pass

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
    @torch.no_grad()
    def step(self, sample: float) -> Optional[Tuple[str, Dict[str, float]]]:
        # 1. O(1) Normalization
        x_norm = self.normalizer.update(sample)
        
        # 2. Stateful SSM + Pooling Step (Now O(1) internally)
        xt = torch.tensor([[x_norm]], device=self.device, dtype=torch.float32)
        logits, self.states = self.model.step(xt, self.states)
        
        # 3. Handle Prediction (Wait for window to fill)
        if len(self.normalizer.samples) < self.window_size:
            return None

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
