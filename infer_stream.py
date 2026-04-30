"""
ONNX inference and MQTT streaming for the Raspberry Pi ECG app.

The deployment path intentionally avoids PyTorch. It requires onnxruntime,
paho-mqtt, and numpy.
"""

import json
import math
import struct
import time
from collections import deque
from pathlib import Path
from queue import Queue
from typing import Dict, Generator, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


LABEL_MAP = {
    0: "N (Normal)",
    1: "S (SVEB)",
    2: "V (VEB)",
    3: "F (Fusion)",
    4: "Q (Unknown)",
}

WARMUP_SAMPLES = 187


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
    """O(1) sliding-window mean/std normalizer."""

    def __init__(self, window_size: int = WARMUP_SAMPLES):
        self.window_size = window_size
        self.samples = deque(maxlen=window_size)
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0

    def update(self, x: float) -> float:
        self.count += 1

        if self.count % 100_000 == 0 and len(self.samples) == self.window_size:
            arr = np.array(self.samples)
            self.mean = float(arr.mean())
            self.m2 = float(((arr - self.mean) ** 2).sum())

        if len(self.samples) < self.window_size:
            self.samples.append(x)
            old_mean = self.mean
            self.mean += (x - old_mean) / len(self.samples)
            self.m2 += (x - old_mean) * (x - self.mean)
        else:
            x_old = self.samples.popleft()
            self.samples.append(x)
            old_mean = self.mean
            new_mean = old_mean + (x - x_old) / self.window_size
            self.mean = new_mean
            self.m2 += (x - x_old) * (x - new_mean + x_old - old_mean)
            self.m2 = max(0.0, self.m2)

        n = len(self.samples)
        std = math.sqrt(self.m2 / n) if n > 0 else 1.0
        std = std if std > 1e-6 else 1.0
        return (x - self.mean) / std


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


class ONNXRealtimeRunner:
    def __init__(self, models_dir: Path):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")

        model_path = Path(models_dir) / "ecg_ssm.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.output_names = [out.name for out in self.session.get_outputs()]

        inputs = self.session.get_inputs()
        input_names = [inp.name for inp in inputs]
        self.input_name = "input" if "input" in input_names else input_names[0]
        self.input_shape = next(inp.shape for inp in inputs if inp.name == self.input_name)

        self.window_size = WARMUP_SAMPLES
        self.normalizer = WelfordOnline(window_size=self.window_size)
        self.window = deque(maxlen=self.window_size)

        self.states: Dict[str, np.ndarray] = {}
        for inp in inputs:
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
        x_norm = self.normalizer.update(sample)
        self.window.append(x_norm)
        if len(self.window) < self.window_size:
            return None

        feeds = {self.input_name: self._format_input(x_norm)}
        feeds.update(self.states)

        outputs = self.session.run(self.output_names, feeds)
        logits = np.asarray(outputs[0], dtype=np.float32)

        if self.states:
            state_input_names = list(self.states.keys())
            for i, out_meta in enumerate(self.session.get_outputs()[1:]):
                state_name = out_meta.name[:-4] if out_meta.name.endswith("_out") else out_meta.name
                if state_name not in self.states and i < len(state_input_names):
                    state_name = state_input_names[i]
                self.states[state_name] = outputs[i + 1]

        probs = softmax_np(logits.ravel())
        pred = int(np.argmax(probs))
        label = LABEL_MAP.get(pred, str(pred))
        return label, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}


def parse_mqtt_samples(payload: str) -> list[float]:
    payload = payload.strip()
    if not payload:
        return []

    try:
        return [float(payload)]
    except ValueError:
        pass

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        try:
            return [float(v) for v in data["samples"]]
        except (TypeError, ValueError):
            return []

    if isinstance(data, list):
        try:
            return [float(v) for v in data]
        except (TypeError, ValueError):
            return []

    if "," in payload:
        try:
            return [float(part.strip()) for part in payload.split(",") if part.strip()]
        except ValueError:
            return []

    return []


def mqtt_stream(
    host: str,
    port: int = 1883,
    topic: str = "ecg/data",
    keepalive: int = 60,
) -> Generator[float, None, None]:
    """Yield individual ECG samples received over MQTT."""
    if mqtt is None:
        raise RuntimeError("paho-mqtt not installed. Run: pip install paho-mqtt")

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
            _, seq, count = struct.unpack("<BIB", payload[:6])
            if expected_seq is not None and seq != expected_seq:
                lost = (seq - expected_seq) & 0xFFFFFFFF
                print(f"WARNING: Packet loss detected. Expected {expected_seq}, got {seq}. Lost ~{lost} batches.")
            expected_seq = (seq + 1) & 0xFFFFFFFF

            data_end = 6 + count * 4
            if len(payload) < data_end + 1:
                return
            if crc8_python(payload[:data_end]) != payload[data_end]:
                print("CRC mismatch; packet dropped")
                return
            for sample in struct.unpack(f"<{count}f", payload[6:data_end]):
                samples.put(sample)
        else:
            try:
                line = payload.decode("utf-8", errors="ignore").strip()
                for sample in parse_mqtt_samples(line):
                    samples.put(sample)
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
