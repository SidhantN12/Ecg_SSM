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

**Stability via Tanh Mapping:**
To ensure stability, eigenvalues of the transition matrix $A$ must reside within the unit circle ($|a| < 1$). This is enforced by passing raw parameters through a `tanh` activation function.

### 2.2 ECG Signal Analysis & MIT-BIH Dataset

- **N**: Normal beat
- **S**: Supraventricular ectopic beat (SVEB)
- **V**: Ventricular ectopic beat (VEB)
- **F**: Fusion beat
- **Q**: Unknown/unclassifiable beat

**Normalization (Z-Score):**
The system employs **Z-score normalization per heartbeat**:
$$\hat{x} = \frac{x - \mu}{\sigma}$$
Hardware streams use a true **sliding-window Welford's algorithm** for $O(1)$ rolling normalization. This ensures stable mean and variance tracking without the performance impact of $O(T)$ buffer recalculation.

---

## 3. Codebase Architecture

### `ssm_model.py`
The core neural network definition. Implements `SimpleSSMLayer`, `SSMEncoder`, and `ECGSSMClassifier`. It supports both global FFT-based convolution for training and a stateful `step()` method for inference.

### `train.py`
The training pipeline. Handles dataset fetching (Kaggle), preprocessing, training, and checkpointing.

### `infer_stream.py`
The inference engine. Contains `RealtimeRunner` (PyTorch) and `ONNXRealtimeRunner`. It manages various stream generators: `simulate_stream_from_csv`, `serial_stream`, and `mqtt_stream`.

### `export_onnx.py`
Converts a trained PyTorch model to ONNX. It uses an `ONNXExportWrapper` to replace complex FFT ops with a recurrent loop, ensuring compatibility with standard ONNX runtimes.

### `app.py`
The Streamlit frontend dashboard for live visualization, configuration, and inference control. It uses **NumPy circular buffers** for efficient plotting and supports dynamic adjustment of DSP parameters like the normalization window size.

### `esp32_firmware/`
C++/ESP-IDF firmware for high-fidelity ECG publishing. It utilizes the **ADC continuous (DMA)** driver to eliminate sampling jitter and a **Binary Protocol (Header + Seq + Data + CRC8)** for robust, low-latency telemetry.

---

## 4. Technical Reference

### `ssm_model.py`

#### `SimpleSSMLayer.step(u, state)`
Executes a single-step recurrence for streaming.
- **Input**: `u` (B, in_dim), `state` (B, d_state).
- **Returns**: `y` (B, out_dim), `new_state`.

#### `ECGSSMClassifier.step(x, states)`
Executes a single-step recurrence and updates **Rolling Pooling States** (Mean and Max).
- **Internal States**: Maintains a `RollingPool` object that provides $O(1)$ mean and $O(T*H)$ max for pooling.
- **Output**: Returns the class probabilities based on the current window's latent state summary.

### `infer_stream.py`

#### `mqtt_stream(host, port, topic)`
A generator that connects to an MQTT broker and yields ECG samples. It decodes Architecture v1 **Binary Packets**, verifies the **CRC8 checksum**, and handles sequence tracking to ensure high-fidelity reconstruction.

#### `ONNXRealtimeRunner` (Class)
High-performance inference runner using ONNX Runtime. Unlike the PyTorch runner, it operates on a sliding window buffer rather than stateful recurrence, making it extremely robust.

### `export_onnx.py`

#### `ONNXExportWrapper` (Class)
Re-implements the forward pass using the `step()` method to avoid `torch.fft` ops, which are often poorly supported in edge ONNX runtimes.

---

### Architecture v1 Telemetry
1. Flash `esp32_firmware/` to an ESP32 WROOM 32.
2. Configuration is handled via `idf.py menuconfig`.
3. Select **MQTT** source in `app.py`.
