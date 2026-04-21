# ECG State-Space Model (SSM) — Realtime Demo

This project implements a lightweight **Diagonal State-Space Model (DSSM)** in PyTorch for real-time classification of ECG heartbeats.

## What you get
- **Custom SSM Architecture**: A dependency-free, high-performance diagonal SSM implemented from scratch.
- **Real-time Engine**: Truly stateful inference supporting consistent $O(1)$ state updates.
- **Streamlit Dashboard**: Live visualization of ECG signals and class probability distributions.
- **Multi-Source Support**: Stream data from CSV files, Serial (USB), or **MQTT (WiFi)**.
- **Inference Backends**: Support for both **PyTorch** and **ONNX Runtime** for edge deployment.
## Hardware Architecture

### Architecture v1 (Current)
- **Sensor**: AD8232 Heart Rate Monitor.
- **Edge Device**: ESP32 WROOM 32 (Continuous DMA sampling, Software Biquad Filtering).
- **Host**: Raspberry Pi 4 (MQTT Broker, Streamlit GUI, SSM Inference).

### Architecture v2 (Future)
- **Deterministic Sampler**: RP2040 dedicated to high-precision acquisition.
- **Hardware Filtering**: SLG47910V GreenPAK FPGA for zero-CPU filtering.
- **System Orchestrator**: ESP32 managing telemetry, power states, and OTA updates.

---

## Quick Start

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2) Run Realtime App
```bash
streamlit run app.py
```
By default, the app is configured for **MQTT Input** from the Architecture v1 edge device.

---

## Hardware Integration

For wireless ECG monitoring with Architecture v1 high-fidelity acquisition:
1.  Flash the firmware located in `esp32_firmware/` to your **ESP32 WROOM 32**.
2.  The firmware uses **DMA continuous sampling** at a hardcoded **187 Hz** and publishes via a **Binary Protocol** for zero-jitter telemetry.
3.  In the Streamlit sidebar:
    - Configure your **MQTT Broker Host**.
    - Click **Start** — sample rate and normalization window are fixed system-wide.

---

## Technical Notes
- **Vectorized Training**: Uses an FFT-based global convolution representation for $O(L \log L)$ training efficiency.
- **Stateful Inference**: Maintains persistent latent states and **Rolling Pooling States** (Mean/Max via Monotonic Queue) for true $O(1)$ updates per sample.
- **Stability Mapping**: SSM eigenvalues are constrained via $1 - \text{softplus}(a_{raw})$, guaranteeing positive-real decay and preventing high-frequency oscillation.
- **Normalization**: Implements a drift-resistant **sliding-window Welford's algorithm** with periodic numerical recalibration every 100k samples to prevent `m2` from going negative.
- **Hardcoded 187 Hz**: The system frequency is a compile-time constant across firmware, Python inference, and model exports — ensuring training-inference alignment with the MIT-BIH dataset.
- **High-Integrity Telemetry**: Binary MQTT packets carry **CRC8 verification** and monotonically incrementing sequence numbers; the host detects and logs any gaps.
- **Stateful ONNX Export**: `export_onnx.py` exposes each encoder layer's hidden state as a named Graph Input/Output, enabling true recurrent inference on the RPi 4 via ONNX Runtime.
