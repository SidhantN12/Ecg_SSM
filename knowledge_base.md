# ECG SSM Knowledge Base

This document provides a comprehensive technical reference for the `Ecg_SSM` project, detailing the architecture, theoretical underpinnings, and implementation specifics for real-time ECG classification using State Space Models (SSMs).

---

## 1. Executive Summary

The `Ecg_SSM` project implements a lightweight, dependency-free State Space Model (SSM) in PyTorch to classify ECG heartbeats in real-time. It targets the MIT-BIH Arrhythmia Database, categorizing heartbeats into five distinct classes according to AAMI standards. The system supports:
- **MQTT input** via WiFi (Architecture v1: ESP32).
- **Architecture v2 support** (RP2040 sampler + SLG47910V filtering).
- **Flexible inference** using PyTorch or ONNX Runtime.

---

## 2. Theoretical Foundation

### 2.1 Diagonal State Space Models (DSSMs)

State Space Models bridge the gap between Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs).

**Discrete-Time Recurrence:**
The core of the SSM is the linear recurrence relation:
$$x_{t+1} = A x_t + B u_t$$
$$y_t = C x_t + D u_t$$

Where:
- $u_t$: Input signal at time $t$.
- $x_t$: Latent state representing the "memory" of the system.
- $y_t$: Output signal.
- $A, B, C, D$: Learnable parameters.

**Diagonal Constraint:**
In this implementation, the matrix $A$ is constrained to be **diagonal**. This allows the update to be computed as an element-wise multiplication ($x_{t+1} = a \odot x_t + b \cdot u_t$), which is significantly more efficient than full matrix-vector products.

**Stability via Softplus Mapping:**
To ensure stability and prevent high-frequency oscillation, eigenvalues of the transition matrix $A$ must be positive and within the unit circle ($0 < a < 1$). This is enforced by the mapping $a = 1 - \text{softplus}(a_{raw})$, which bounds $a$ strictly in $(0, 1)$. The previous `tanh` mapping allowed negative eigenvalues, causing the latent state to oscillate at the Nyquist frequency — physically inappropriate for low-frequency ECG signals.

### 2.2 ECG Signal Analysis & MIT-BIH Dataset

- **N**: Normal beat
- **S**: Supraventricular ectopic beat (SVEB)
- **V**: Ventricular ectopic beat (VEB)
- **F**: Fusion beat
- **Q**: Unknown/unclassifiable beat

**Normalization (Z-Score):**
The system employs **Z-score normalization per heartbeat**:
$$\hat{x} = \frac{x - \mu}{\sigma}$$
Hardware streams use a **sliding-window Welford's algorithm** for $O(1)$ rolling normalization, with a drift-resistance mechanism: every 100,000 samples, `mean` and `m2` are recomputed exactly from the buffer contents, preventing floating-point catastrophic cancellation from causing `m2` to go negative and crash the square root.

**System Frequency:**
The entire stack operates at a fixed **187 Hz**, matching the MIT-BIH Arrhythmia Database resampling rate. This constant is hardcoded in `filter_pipeline.hpp`, `infer_stream.py`, and the `Kconfig` (sample rate entry has been removed to prevent misconfiguration).

---

## 3. Codebase Architecture

### `ssm_model.py`
The core neural network definition. Implements `SimpleSSMLayer`, `SSMEncoder`, `ECGSSMClassifier`, `MonotonicQueue`, and `RollingPool`.
- **Training path**: FFT-based global convolution for $O(L \log L)$ efficiency.
- **Inference path**: Stateful `step()` + `step_stateful()` methods for true $O(1)$ per-sample updates.
- **Eigenvalue stability**: Transition matrix $A$ uses $1 - \text{softplus}(a_{raw})$ mapping to guarantee positive-real, stable decay.
- **Pooling**: `RollingPool` uses a per-dimension `MonotonicQueue` for true $O(1)$ sliding max, replacing the old $O(W \times H)$ `torch.max` scan.

### `train.py`
The training pipeline. Handles dataset fetching (Kaggle), preprocessing, training, and checkpointing. Optimized for x86 (Ryzen 5650U): `num_workers=8` default, `pin_memory=True`. Supports `--stitch` for sequential beat ordering to improve stateful stability.

### `infer_stream.py`
The inference engine. Contains `RealtimeRunner` (PyTorch) and `ONNXRealtimeRunner`. Both runners are stateful, operating at a hardcoded **187 Hz** with a fixed 187-sample normalization window.
- `WelfordOnline`: Drift-resistant with periodic recalibration every 100k samples.
- `mqtt_stream`: Tracks MQTT sequence numbers and logs packet loss.

### `export_onnx.py`
Exports the trained model to ONNX in **stateful one-step mode**: the `ONNXExportWrapper` calls `model.step_stateful()` and exposes each encoder layer's hidden state as a named Graph Input and Output. This enables `ONNXRealtimeRunner` to persist state between individual sample calls.

### `app.py`
The Streamlit frontend dashboard. Sample rate (187 Hz) and normalization window (187 samples) are fixed constants — removed from the sidebar to prevent misconfiguration.

### `esp32_firmware/`
C++/ESP-IDF firmware for high-fidelity ECG publishing. Key properties:
- **Sample rate**: Hardcoded to **187 Hz** (`SAMPLE_RATE_HZ = 187`).
- **Packet buffer**: Correctly sized to `sizeof(Header) + (BATCH_SIZE × 4) + 1` bytes.
- **Lead-off detection**: Reads `LO+` / `LO-` GPIOs each acquisition cycle; outputs `0.0f` if either pin is asserted, preventing garbage classification.
- **Sequence tracking**: Host-side sequence counter detects and logs dropped MQTT packets.

---

## 4. Technical Reference

### `ssm_model.py`

#### `SimpleSSMLayer.step(u, state)`
Executes a single-step recurrence for streaming.
- **Input**: `u` (B, in_dim), `state` (B, d_state).
- **Returns**: `y` (B, out_dim), `new_state`.
- **Stability**: Uses $a = 1 - \text{softplus}(a_{raw})$ to constrain eigenvalues to $(0, 1)$.

#### `MonotonicQueue`
Per-dimension sliding max tracking for `RollingPool`. Each dimension maintains a deque of `(value, index)` pairs in monotonically decreasing order.
- **Complexity**: $O(1)$ amortised push and max query, $O(H)$ per timestep where $H$ is hidden dimension.

#### `RollingPool.update(h)`
- **Average**: $O(1)$ via running sum and length.
- **Max**: $O(H)$ via `MonotonicQueue` per dimension — $O(1)$ relative to window size $W$.

#### `ECGSSMClassifier.step(x, states)`
Executes a single-step recurrence and updates `RollingPool`.
- **Output**: Class logits based on avg+max pooled latent state.

#### `ECGSSMClassifier.step_stateful(x, encoder_states)`
ONNX-compatible single-step encoder. Takes a flat list of layer states, returns updated hidden representation and new states. Used exclusively by `export_onnx.py`.

### `infer_stream.py`

#### `WelfordOnline`
Sliding-window online normalizer. Fixed window of 187 samples.
- **Drift prevention**: Recomputes `mean` and `m2` exactly from buffer every 100,000 samples.
- **Safety**: `m2` is clamped to `max(0.0, m2)` before square root.

#### `mqtt_stream(host, port, topic)`
A generator that connects to an MQTT broker and yields ECG samples. Decodes Architecture v1 **Binary Packets**, verifies **CRC8**, and tracks sequence numbers — printing a warning on any gap.

#### `RealtimeRunner`
Primary stateful inference runner using PyTorch. Fixed 187 Hz / 187-sample window. Persistent SSM hidden states are carried across the entire stream.

#### `ONNXRealtimeRunner`
Stateful inference runner using ONNX Runtime. Inspects the exported graph at load time to discover all state inputs. Maintains state tensors between calls, providing equivalent behaviour to `RealtimeRunner`.

### `export_onnx.py`

#### `ONNXExportWrapper`
Wraps `model.step_stateful()`. The `forward(x, *states)` signature takes a single sample and a variable list of per-layer state tensors, returning logits and updated states. This makes the ONNX graph fully recurrent with named, persistent I/O.

---

### Architecture v1 Telemetry
1. Flash `esp32_firmware/` to an ESP32 WROOM 32.
2. Configuration is handled via `idf.py menuconfig` (WiFi, MQTT URI, Batch Size only — sample rate is hardcoded).
3. Set **MQTT Source** in `app.py` sidebar and click Start.
