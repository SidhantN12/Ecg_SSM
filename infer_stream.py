"""
infer_stream.py  —  ONNX-only inference + MQTT streaming.
No PyTorch dependency. Requires: onnxruntime, paho-mqtt, numpy.
"""

import time
import math
import json
import struct
from collections import deque
from queue import Queue
from pathlib import Path
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP = {
    0: "N (Normal)",
    1: "S (SVEB)",
    2: "V (VEB)",
    3: "F (Fusion)",
    4: "Q (Unknown)",
}

WARMUP_SAMPLES = 187  # must match training / export constant

# ---------------------------------------------------------------------------
# CRC helper (matches ESP32 firmware)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Online normalizer — drift-resistant sliding-window Welford's algorithm
# ---------------------------------------------------------------------------

class WelfordOnline:
    """O(1) sliding-window mean/std normalizer."""

    def __init__(self, window_size: int = 187):
        self.window_size = window_size
        self.samples: deque = deque(maxlen=window_size)
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0

    def update(self, x: float) -> float:
        self.count += 1

        # Periodic numerical recalibration every 100k samples
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

# ---------------------------------------------------------------------------
# Softmax (numpy)
# ---------------------------------------------------------------------------

def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()

# ---------------------------------------------------------------------------
# ONNX Realtime Runner
# ---------------------------------------------------------------------------

class ONNXRealtimeRunner:
    """
    Stateful, sample-by-sample inference using the exported ONNX graph.

    The ONNX graph (from export_onnx.py) has:
        Inputs : "input", "state_0", "state_1", ..., "state_{N-1}"
        Outputs: "output", "state_0_out", "state_1_out", ..., "state_{N-1}_out"

    On each call to step() we feed the current states in, read the updated
    states out, and store them under the INPUT names ready for the next call.
    """

    def __init__(self, models_dir: Path):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            )

        model_path = Path(models_dir) / "ecg_ssm.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run `python export_onnx.py` to generate it."
            )

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        # Collect input / output metadata once
        self._input_names  = [inp.name for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]

        # "input" is the signal; everything else is a recurrent state
        self._signal_input = "input"
        self._state_input_names  = [n for n in self._input_names  if n != "input"]
        # The corresponding outputs carry an "_out" suffix (from export_onnx.py)
        # e.g. state_0 → state_0_out.  We build a mapping: output_name → input_name
        self._out_to_in: Dict[str, str] = {}
        for in_name in self._state_input_names:
            out_name = in_name + "_out"
            if out_name in self._output_names:
                self._out_to_in[out_name] = in_name
            else:
                # Fallback: same name (defensive)
                self._out_to_in[in_name] = in_name

        # Initialize zero states keyed by INPUT name
        self._states: Dict[str, np.ndarray] = {}
        for inp in self.session.get_inputs():
            if inp.name == "input":
                continue
            shape = [1 if (isinstance(s, str) or s < 0) else s for s in inp.shape]
            self._states[inp.name] = np.zeros(shape, dtype=np.float32)

        self.normalizer = WelfordOnline(window_size=WARMUP_SAMPLES)

    # ------------------------------------------------------------------

    def step(self, sample: float) -> Optional[Tuple[str, Dict[str, float]]]:
        """
        Process one sample.  Returns (label, probs_dict) after warm-up,
        None during warm-up.
        """
        x_norm = self.normalizer.update(sample)

        if self.normalizer.count < WARMUP_SAMPLES:
            return None

        # Build feed dict: signal + current states (keyed by INPUT name)
        feed: Dict[str, np.ndarray] = {
            self._signal_input: np.array([[x_norm]], dtype=np.float32)
        }
        feed.update(self._states)

        outputs = self.session.run(self._output_names, feed)
        out_map = dict(zip(self._output_names, outputs))

        # Update states: map output names → input names
        for out_name, in_name in self._out_to_in.items():
            self._states[in_name] = out_map[out_name]

        # First output is logits
        logits = np.asarray(out_map[self._output_names[0]], dtype=np.float32).ravel()
        probs  = softmax_np(logits)
        pred   = int(np.argmax(probs))
        label  = LABEL_MAP.get(pred, str(pred))

        return label, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}

# ---------------------------------------------------------------------------
# MQTT streaming generator
# ---------------------------------------------------------------------------

def parse_mqtt_samples(payload: str) -> list:
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
            return [float(p.strip()) for p in payload.split(",") if p.strip()]
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

    samples: Queue = Queue()
    errors:  Queue = Queue()
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
            # Binary telemetry protocol
            magic, seq, count = struct.unpack("<BIB", payload[:6])
            if expected_seq is not None and seq != expected_seq:
                lost = (seq - expected_seq) & 0xFFFFFFFF
                print(
                    f"WARNING: Packet Loss Detected! "
                    f"Expected {expected_seq}, got {seq}. Lost ~{lost} batches."
                )
            expected_seq = (seq + 1) & 0xFFFFFFFF

            data_end = 6 + count * 4
            if len(payload) < data_end + 1:
                return
            if crc8_python(payload[:data_end]) != payload[data_end]:
                print("CRC Mismatch — packet dropped")
                return
            for s in struct.unpack(f"<{count}f", payload[6:data_end]):
                samples.put(s)
        else:
            # Fallback: UTF-8 text / JSON / CSV
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