import time
from collections import deque
from queue import Queue
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple
import json
import struct
import math

import numpy as np

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

try:
    import torch
except Exception:
    torch = None


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
        self.count = 0

    def update(self, x: float) -> float:
        self.count += 1
        # Periodic recalibration to prevent numerical drift (every 100k samples)
        if self.count % 100000 == 0 and len(self.samples) == self.window_size:
            arr = np.array(self.samples)
            self.mean = arr.mean()
            self.m2 = ((arr - self.mean)**2).sum()

        if len(self.samples) < self.window_size:
            self.samples.append(x)
            old_mean = self.mean
            self.mean += (x - old_mean) / len(self.samples)
            self.m2 += (x - old_mean) * (x - self.mean)
        else:
            x_old = self.samples.popleft()
            self.samples.append(x)
            
            # Robust sliding update
            old_mean = self.mean
            new_mean = old_mean + (x - x_old) / self.window_size
            self.mean = new_mean
            self.m2 += (x - x_old) * (x - new_mean + x_old - old_mean)

        self.m2 = max(0.0, self.m2) # Safety against negative m2
        std = math.sqrt(self.m2 / len(self.samples)) if len(self.samples) > 0 else 1.0
        std = std if std > 1e-6 else 1.0
        return (x - self.mean) / std


def load_model(models_dir: Path) -> object:
    if torch is None:
        raise RuntimeError("torch not installed. Install torch to use the PyTorch runner, or use ONNXRealtimeRunner.")

    from ssm_model import ECGSSMClassifier, LegacyECGSSMClassifier

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
    import pandas as pd

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

    expected_seq = None

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc != 0:
            errors.put(RuntimeError(f"MQTT connection failed with code {rc}"))
            return
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        nonlocal expected_seq
        payload = msg.payload
        if not payload or len(payload) < 7:
            return
        
        if payload[0] == 0xA5:
            header_size = 6
            magic, seq, count = struct.unpack("<BIB", payload[:6])
            
            # Sequence Tracking
            if expected_seq is not None:
                if seq != expected_seq:
                    lost = (seq - expected_seq) & 0xFFFFFFFF
                    print(f"WARNING: Packet Loss Detected! Expected {expected_seq}, got {seq}. Lost ~{lost} batches.")
            expected_seq = (seq + 1) & 0xFFFFFFFF

            data_end = header_size + count * 4
            if len(payload) < data_end + 1:
                return
            
            received_crc = payload[data_end]
            calculated_crc = crc8_python(payload[:data_end])
            if received_crc != calculated_crc:
                print(f"CRC Mismatch: {received_crc} != {calculated_crc}")
                return
            
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
    def __init__(self, models_dir: Path, device: Optional[str] = None):
        if torch is None:
            raise RuntimeError("torch not installed. Install torch to use RealtimeRunner, or use ONNXRealtimeRunner.")

        self.model = load_model(models_dir)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model.to(self.device)
        self.window_size = 187 # Fixed for standardize flow
        self.normalizer = WelfordOnline(window_size=self.window_size)
        self.states = None

    def step(self, sample: float) -> Optional[Tuple[str, Dict[str, float]]]:
        if torch is None:
            raise RuntimeError("torch not installed. Install torch to use RealtimeRunner, or use ONNXRealtimeRunner.")

        with torch.no_grad():
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
    def __init__(self, models_dir: Path):
        if ort is None:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime")

        model_path = Path(models_dir) / "ecg_ssm.onnx"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        input_names = [inp.name for inp in self.session.get_inputs()]
        self.input_name = "input" if "input" in input_names else input_names[0]
        self.input_shape = next(inp.shape for inp in self.session.get_inputs() if inp.name == self.input_name)
        self.window_size = 187
        self.normalizer = WelfordOnline(window_size=self.window_size)
        self.window = deque(maxlen=self.window_size)

        self.states = {}
        for inp in self.session.get_inputs():
            if inp.name == self.input_name:
                continue
            self.states[inp.name] = np.zeros(self._static_shape(inp.shape), dtype=np.float32)

    @staticmethod
    def _static_shape(shape) -> list[int]:
        return [dim if isinstance(dim, int) and dim > 0 else 1 for dim in shape]

    def _uses_window_input(self) -> bool:
        return any(dim == self.window_size for dim in self.input_shape if isinstance(dim, int))

    def _format_input(self, sample: float) -> np.ndarray:
        if not self._uses_window_input():
            return np.array([[sample]], dtype=np.float32)

        x = np.asarray(self.window, dtype=np.float32)
        if len(self.input_shape) == 3:
            if self.input_shape[1] == self.window_size:
                return x.reshape(1, self.window_size, 1)
            return x.reshape(1, 1, self.window_size)
        return x.reshape(1, self.window_size)

    def step(self, sample: float) -> Optional[Tuple[str, Dict[str, float]]]:
        # 1. O(1) Normalization
        x_norm = self.normalizer.update(sample)
        self.window.append(x_norm)
        if len(self.window) < self.window_size:
            return None

        # 2. ONNX inference
        inputs = {self.input_name: self._format_input(x_norm)}
        inputs.update(self.states)
        
        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        
        if self.states:
            # Update states from outputs. Export names are state_0_out, state_1_out, ...
            state_input_names = [name for name in self.states.keys()]
            for i, out_meta in enumerate(self.session.get_outputs()[1:]):
                state_name = out_meta.name[:-4] if out_meta.name.endswith("_out") else out_meta.name
                if state_name not in self.states and i < len(state_input_names):
                    state_name = state_input_names[i]
                self.states[state_name] = outputs[i + 1]

        probs = softmax_np(np.asarray(logits[0], dtype=np.float32))
        pred = int(np.argmax(probs))
        label = LABEL_MAP.get(pred, str(pred))
        return label, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
