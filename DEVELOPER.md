# ECG Architecture v1 Developer Guide

This document provides comprehensive instructions for setting up and running the ECG State-Space Model project using **Architecture v1**.

## 1. System Overview (Architecture v1)

*   **Sensor**: AD8232 Heart Rate Monitor (Analog).
*   **Edge Device**: ESP32 (Continuous ADC sampling via DMA, real-time Biquad filtering, Binary MQTT publishing).
*   **Host**: Raspberry Pi 4 (MQTT Broker, Streamlit Engine, SSM Inference).

---

## 2. Prerequisites

### Hardware
*   1x ESP32 WROOM 32 Development Board.
*   1x AD8232 ECG Module + Electrodes.
*   1x Raspberry Pi 4 (or PC acting as the host).
*   Jumper wires and a breadboard.

### Software
*   **On Raspberry Pi / Host**:
    *   Python 3.9+
    *   MQTT Broker: [Mosquitto](https://mosquitto.org/)
*   **For ESP32 Development**:
    *   [ESP-IDF v5.1+](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html)

---

## 3. Ground Station Setup (Raspberry Pi)

### 3.1 Install MQTT Broker
On the Raspberry Pi, install and start the Mosquitto broker:
```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```
*Note: By default, Mosquitto listens on port 1883. Ensure your firewall allows this.*

### 3.2 Install Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Edge Device Setup (ESP32)

### 4.1 Wiring Instructions
| AD8232 Pin | ESP32 Pin | Description |
|------------|-----------|-------------|
| 3.3V       | 3.3V      | Power       |
| GND        | GND       | Ground      |
| OUTPUT     | GPIO34    | Analog Signal (ADC1_CH6) |
| LO+        | GPIO25    | Lead-off Positive |
| LO-        | GPIO26    | Lead-off Negative |

### 4.2 Configure Firmware
Navigate to the `esp32_firmware` directory and open the configuration menu:
```bash
cd esp32_firmware
idf.py menuconfig
```
Set the following under **ECG Project Configuration**:
*   **WiFi SSID & Password**: Your network credentials.
*   **MQTT Broker URI**: `mqtt://<YOUR_RPI_IP_ADDRESS>`
*   **Batch Size**: Number of samples per MQTT packet (default 8).

> **Note:** The sample rate is hardcoded to **187 Hz** in firmware and is not configurable via menuconfig. This aligns with the MIT-BIH dataset frequency.

### 4.3 Build and Flash
```bash
idf.py set-target esp32
idf.py build
idf.py -p <PORT> flash monitor
```

---

## 5. Running the Application

1.  Ensure the ESP32 is powered and connected to WiFi (check terminal monitor).
2.  On the Raspberry Pi, run the WebGUI:
    ```bash
    streamlit run app.py
    ```
3.  Open the URL provided by Streamlit in your browser.
4.  In the sidebar, enter the **MQTT Broker Host** and click **Start**.

> **Note:** Sample rate and normalization window are fixed at **187 Hz / 187 samples** system-wide and are no longer configurable in the sidebar.

---

## 6. Model Lifecycle

### 6.1 Training (Optimized for Ryzen 5650U x86)

Training is designed to be run on the development machine (Ryzen 5650U or equivalent x86). The RPi 4 is inference-only.

1.  Download the dataset (requires Kaggle credentials):
    ```bash
    python train.py --auto-download
    ```
    Or place `mitbih_train.csv` / `mitbih_test.csv` manually inside `data/` (or any subdirectory).

2.  Run full training with x86-optimized parallelism:
    ```bash
    python train.py --num-workers 12 --epochs 50
    ```
    - `--num-workers 12`: Saturates all 12 threads of the 5650U for DataLoader throughput.
    - `pin_memory=True` is enabled automatically for faster CPU ↔ memory transfer.

3.  **Optional — Stitched Training** (improves stateful inference stability):
    ```bash
    python train.py --num-workers 12 --stitch --epochs 50
    ```
    - `--stitch`: Disables dataset shuffling and trains on sequentially ordered beats, helping the SSM learn to carry state across beat boundaries as it does at runtime.

4.  The best checkpoint is saved to `models/ecg_ssm.pt` automatically via early stopping on macro-F1.

**Key CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 5 | Training epochs |
| `--batch-size` | 256 | Batch size |
| `--lr` | 3e-3 | Learning rate |
| `--d-state` | 64 | SSM state dimension |
| `--hidden` | 64 | Hidden layer width |
| `--depth` | 3 | Number of SSM layers |
| `--num-workers` | 8 | DataLoader workers |
| `--stitch` | off | Sequential beat ordering |
| `--max-test` | 10000 | Limit test samples (0 = all) |
| `--auto-download` | off | Auto-download from Kaggle |

### 6.2 ONNX Export & RPi 4 Deployment

For low-latency stateful inference on the Raspberry Pi 4:

1.  Export the trained model:
    ```bash
    python export_onnx.py
    ```
    This wraps the model's `step_stateful()` method, exposing each encoder layer's hidden state as a named Graph Input and Output. This enables the ONNX runtime to be truly recurrent.

2.  The resulting `models/ecg_ssm.onnx` is self-describing — `ONNXRealtimeRunner` inspects its inputs at load time and initialises all state tensors automatically. No manual configuration is required.

3.  Copy `models/ecg_ssm.onnx` to the RPi 4 and run:
    ```bash
    streamlit run app.py
    ```
    The runner will automatically use `ONNXRealtimeRunner` if the `.onnx` file is present and `onnxruntime` is installed.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No predictions for first ~187 samples | Normalizer filling warm-up window | Expected behaviour; wait for window to fill |
| `WARNING: Packet Loss Detected` in terminal | WiFi congestion or broker overload | Check WiFi signal; reduce MQTT QoS or broker load |
| Dashboard shows a sustained flatline | Electrodes disconnected (lead-off) | Re-attach electrodes; firmware outputs `0.0` on LO+ or LO- assertion |
| `CRC Mismatch` errors | Corrupted MQTT payload | Usually transient; check broker and network MTU |
| Inference accuracy drops after reboot | Mismatched `.onnx` file | Re-export with `python export_onnx.py` after retraining |

> **187 Hz is a system invariant.** Do not attempt to change the sample rate in `menuconfig`, `app.py`, or `filter_pipeline.hpp`. All three layers are hardcoded and calibrated to 187 Hz.

---

## 8. Future Scope: Architecture v2

The Architecture v1 refactor has modularized the acquisition and signal processing layers to simplify the **Architecture v2** migration:

### 8.1 Modular Acquisition (RP2040)
*   The ESP32's `adc_continuous` driver will be replaced by a dedicated **RP2040** module.
*   Because acquisition is decoupled from transport in `main.cpp`, the ESP32 will simply swap the DMA reader task for an SPI/UART listener task while the rest of the telemetry pipeline remains unchanged.

### 8.2 Hardware Filtering (SLG47910V)
*   The `EcgFilterPipeline` class in `filter_pipeline.hpp` serves as a software wrapper.
*   In v2, this wrapper will be updated to interface with the **SLG47910V GreenPAK FPGA** via I2C/SPI, allowing the CPU to bypass software biquads entirely while maintaining the same `apply()` interface for the system.

### 8.3 System Orchestration (ESP32)
*   The ESP32's role transitions to a **System Orchestrator**, managing the lifecycle of edge components and handling secure WiFi/BLE communication.
