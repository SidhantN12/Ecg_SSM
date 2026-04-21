# ECG State-Space Model (SSM) — Realtime Demo

This project implements a lightweight **Diagonal State-Space Model (DSSM)** in PyTorch for real-time classification of ECG heartbeats.

## What you get
- **Custom SSM Architecture**: A dependency-free, high-performance diagonal SSM implemented from scratch.
- **Real-time Engine**: Truly stateful inference supporting consistent $O(1)$ state updates.
- **Streamlit Dashboard**: Live visualization of ECG signals and class probability distributions.
- **Multi-Source Support**: Stream data from CSV files, Serial (USB), or **MQTT (WiFi)**.
- **Inference Backends**: Support for both **PyTorch** and **ONNX Runtime** for edge deployment.
- **Hardware Support**: Ready for AD8232 sensors via Serial or ESP32-S3 (via MQTT).

---

## Theoretical Foundation

### Diagonal State Space Models (DSSMs)
State Space Models bridge the gap between RNNs and CNNs. The core recurrence is:
$$x_{t+1} = A x_t + B u_t$$
$$y_t = C x_t + D u_t$$

In this project, $A$ is constrained to be **diagonal**, enabling element-wise updates that are significantly faster than dense matrix operations. Stability is guaranteed by constraining eigenvalues via a `tanh` mapping ($|a| < 1$).

### Heartbeat Classification (AAMI Standards)
The model classifies heartbeats into the five standard AAMI categories based on the MIT-BIH Arrhythmia Database:
- **N**: Normal beat
- **S**: Supraventricular ectopic beat (SVEB)
- **V**: Ventricular ectopic beat (VEB)
- **F**: Fusion beat
- **Q**: Unknown/unclassifiable beat

---

## Quick Start

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2) Download Data & Train
The dataset is hosted on Kaggle. You must have your Kaggle API credentials exported as environment variables for auto-download to work.

```bash
# Run training with the auto-download flag
python train.py --epochs 10 --batch-size 128 --auto-download
```

### 3) Export to ONNX (Optional)
For faster inference or deployment to non-Python environments:
```bash
python export_onnx.py
```

### 4) Run Realtime App
```bash
streamlit run app.py
```
By default, the app runs in **Simulated** mode using `mitbih_test.csv`.

---

## Hardware Integration

### Serial Input (Arduino/AD8232)
To use a live sensor (e.g., AD8232 + Arduino):
1. Upload this sketch to your Arduino:
   ```cpp
   void setup() { Serial.begin(115200); }
   void loop() { Serial.println(analogRead(A0)); delay(4); } // ~250Hz sampling
   ```
2. In the Streamlit sidebar:
   - Select **Input source: Serial**.
   - Set the correct **COM port** (e.g., `/dev/ttyUSB0` or `COM3`).

### MQTT Input (ESP32-S3 / WiFi)
For wireless ECG monitoring:
1. Flash the firmware located in `esp32_firmware/` to your ESP32-S3.
2. The firmware publishes to `ecg/data` by default.
3. In the Streamlit sidebar:
   - Select **Input source: MQTT**.
   - Configure your **MQTT Broker Host** (e.g., `localhost` or a dedicated IP).

---

## Technical Notes
- **Vectorized Training**: Uses an FFT-based global convolution representation for $O(L \log L)$ training efficiency.
- **Stateful Inference**: Maintains persistent latent states for $O(1)$ updates per sample during streaming.
- **Normalization**: Implements **Welford's online algorithm** for $O(1)$ rolling Z-score normalization.
- **ONNX Optimization**: The `export_onnx.py` script replaces complex-valued FFT operations with a recurrent `step()` loop for maximum compatibility with edge inference engines.
