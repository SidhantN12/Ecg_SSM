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
2.  The firmware uses **DMA continuous sampling** and publishes via a **Binary Protocol** for zero-jitter telemetry.
3.  In the Streamlit sidebar:
    - Select **Input source: MQTT**.
    - Configure your **MQTT Broker Host**.
    - Adjust **Normalization Window** (samples) for optimal Z-score tracking.

---

## Technical Notes
- **Vectorized Training**: Uses an FFT-based global convolution representation for $O(L \log L)$ training efficiency.
- **Stateful Inference**: Maintains persistent latent states and **Rolling Pooling States** (Mean/Max) for true $O(1)$ updates per sample.
- **Normalization**: Implements a true **$O(1)$ sliding-window Welford's algorithm** for robust rolling Z-score normalization.
- **High-Integrity Telemetry**: The MQTT protocol utilizes a binary packet format with **CRC8 verification** and sequence numbering to handle packet loss and ensure data integrity.
- **ONNX Optimization**: The `export_onnx.py` script replaces complex-valued FFT operations with a recurrent `step()` loop for maximum compatibility with edge inference engines.
