# ECG SSM: Real-Time ECG Classification with MQTT, ONNX, and ESP32-S3

This project implements a lightweight ECG heartbeat classifier based on a diagonal State Space Model (SSM). It supports:

- offline training in PyTorch
- ONNX export for lightweight edge inference
- a Streamlit dashboard for real-time visualization
- simulated CSV streaming
- serial streaming
- MQTT streaming from an ESP32-S3 + AD8232 pipeline

The current production-style architecture is:

`AD8232 -> ESP32-S3 -> WiFi/MQTT -> Raspberry Pi or laptop -> ONNX inference -> Streamlit dashboard`

## 1. What This Repo Contains

Core files:

- [ssm_model.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/ssm_model.py:1): SSM model definitions
- [train.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/train.py:1): training pipeline for MIT-BIH heartbeat classification
- [export_onnx.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/export_onnx.py:1): ONNX export path
- [infer_stream.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/infer_stream.py:1): realtime runners and input stream generators
- [app.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/app.py:1): Streamlit dashboard
- [esp32_firmware/main/main.cpp](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/esp32_firmware/main/main.cpp:1): ESP32-S3 MQTT publisher firmware
- [esp32_firmware/Kconfig.projbuild](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/esp32_firmware/Kconfig.projbuild:1): firmware configuration via `menuconfig`
- [knowledge_base.md](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/knowledge_base.md:1): project background and theory

## 2. Model Summary

The model is a diagonal SSM for ECG heartbeat classification.

Heartbeat classes:

- `0`: `N (Normal)`
- `1`: `S (SVEB)`
- `2`: `V (VEB)`
- `3`: `F (Fusion)`
- `4`: `Q (Unknown)`

Input conventions:

- training input shape: `(batch, time)`
- typical heartbeat window: `187` samples
- realtime ONNX inference input shape: `(1, 187)`

Important implementation note:

- the training `forward()` path uses FFT-based sequence computation
- the ONNX export path uses an export-friendly recurrent wrapper built on `model.step()`
- the loader supports both the current classifier head and an older legacy checkpoint format already present in `models/ecg_ssm.pt`

## 3. System Architecture

### A. Development / Local Test Mode

`Fake MQTT publisher or CSV -> Mosquitto -> Streamlit app -> ONNX runner`

Use this when:

- you do not have hardware yet
- you want to validate the dashboard
- you want to test MQTT and ONNX locally on your laptop

### B. Edge Deployment Mode

`AD8232 -> ESP32-S3 -> Mosquitto broker on Raspberry Pi -> Python app on Raspberry Pi -> ONNX Runtime -> Streamlit`

Use this when:

- you have the hardware
- you want inference on the Pi
- you want the ESP32 only to collect and publish raw ECG samples

## 4. Why WiFi + MQTT

This repo is designed around continuous sequential ECG samples and rolling windows. WiFi + MQTT is the best fit because it provides:

- low latency on a local network
- higher throughput than needed for ECG
- straightforward integration with Python
- clean compatibility with Raspberry Pi
- ordered and reliable delivery when using standard MQTT/TCP transport

Bluetooth and LoRa-style mesh systems are possible in theory, but they are a worse fit for this project’s realtime windowed inference path.

## 5. Environment Setup

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Python dependencies include:

- `torch`
- `streamlit`
- `plotly`
- `paho-mqtt`
- `onnx`
- `onnxruntime`
- `onnxscript`

### MQTT broker on macOS

If you are developing on macOS:

```bash
brew install mosquitto
cp /opt/homebrew/etc/mosquitto/mosquitto.conf.example /opt/homebrew/etc/mosquitto/mosquitto.conf
brew services start mosquitto
```

Broker test:

Terminal 1:

```bash
mosquitto_sub -h localhost -t ecg/test
```

Terminal 2:

```bash
mosquitto_pub -h localhost -t ecg/test -m "hello"
```

### MQTT broker on Raspberry Pi

If you are running on Raspberry Pi OS:

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients -y
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

## 6. Dataset and Training

This repo targets the MIT-BIH heartbeat dataset from Kaggle.

Train the model:

```bash
python train.py --epochs 10 --batch-size 128 --auto-download
```

Expected output:

- checkpoint file: `models/ecg_ssm.pt`

Training notes:

- `train.py` normalizes each example independently with z-score normalization
- the best checkpoint is saved based on validation F1
- the loader in `infer_stream.py` is checkpoint-format aware

## 7. ONNX Export

Export the trained model:

```bash
python export_onnx.py
```

Expected output:

- `models/ecg_ssm.onnx`

Why the export path is special:

- direct export of the FFT-heavy training graph is not reliable for this model
- `export_onnx.py` wraps the model and replays it step-by-step using the recurrent path
- this keeps the exported graph real-valued and compatible with `onnxruntime`

Quick validation:

```bash
python - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession('models/ecg_ssm.onnx', providers=['CPUExecutionProvider'])
print(sess.get_inputs()[0].shape, sess.get_outputs()[0].shape)
PY
```

Expected shape:

- input: `[1, 187]`
- output: `[1, 5]`

## 8. Running the Dashboard

Start Streamlit:

```bash
streamlit run app.py
```

Available input sources in the UI:

- `Simulated`
- `Serial`
- `MQTT`

Available inference backends:

- `PyTorch`
- `ONNX`

Recommended choices:

- local testing: `MQTT + ONNX`
- Raspberry Pi deployment: `MQTT + ONNX`

## 9. Local MQTT Test Without Hardware

This is the simplest end-to-end validation path.

### Step 1: export ONNX

```bash
python export_onnx.py
```

### Step 2: run the app

```bash
streamlit run app.py
```

### Step 3: configure the sidebar

Set:

- `Input source` = `MQTT`
- `Inference backend` = `ONNX`
- `MQTT broker host` = `localhost`
- `MQTT broker port` = `1883`
- `MQTT topic` = `ecg/data`

### Step 4: publish fake data

Single-sample mode:

```bash
while true; do mosquitto_pub -h localhost -t ecg/data -m "$((RANDOM % 200 + 400))"; sleep 0.01; done
```

Batch mode:

```bash
while true; do mosquitto_pub -h localhost -t ecg/data -m "500,502,498,503,507,509,506,504"; sleep 0.05; done
```

Expected behavior:

- the graph should update continuously
- predictions should start after the rolling window fills with `187` samples

## 10. MQTT Payload Formats Supported by the App

The Python subscriber in [infer_stream.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/infer_stream.py:1) accepts all of the following:

Single sample:

```text
512
```

Comma-separated batch:

```text
512,515,498,502
```

JSON list:

```json
[512, 515, 498, 502]
```

JSON object:

```json
{"samples": [512, 515, 498, 502]}
```

This makes the laptop/Pi side tolerant to future firmware changes.

## 11. Realtime Runners

There are two realtime inference paths.

### `RealtimeRunner`

Defined in [infer_stream.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/infer_stream.py:195).

Use when:

- you want PyTorch inference directly from the checkpoint

Behavior:

- updates running normalization
- uses the model `step()` function
- maintains rolling hidden states for prediction

### `ONNXRealtimeRunner`

Defined in [infer_stream.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/infer_stream.py:218).

Use when:

- you want lightweight deployment
- you want Raspberry Pi inference
- you want the exact exported ONNX model path

Behavior:

- stores a rolling raw-sample window
- normalizes each full window
- runs ONNX on the full `(1, 187)` input
- returns label and probabilities

## 12. ESP32-S3 Firmware

The ESP32 code is in:

- [esp32_firmware/main/main.cpp](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/esp32_firmware/main/main.cpp:1)
- [esp32_firmware/Kconfig.projbuild](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/esp32_firmware/Kconfig.projbuild:1)

What it does:

- connects to WiFi in station mode
- connects to the configured MQTT broker
- reads AD8232 analog samples through ADC1
- optionally checks `LO+` and `LO-` lead-off pins
- publishes raw ECG samples to MQTT
- supports batching multiple samples per publish

### Firmware configuration

From the firmware directory:

```bash
cd esp32_firmware
idf.py menuconfig
```

Open `ECG MQTT Publisher` and configure:

- `WiFi SSID`
- `WiFi password`
- `MQTT broker URI`
- `MQTT publish topic`
- `MQTT QoS`
- `MQTT samples per publish`
- `Sampling rate (Hz)`
- `ADC1 channel for AD8232 output`
- `ADC attenuation enum`
- optional `LO+` and `LO-` GPIOs

Recommended defaults:

- topic: `ecg/data`
- QoS: `0`
- samples per publish: `8`
- sample rate: `250`
- attenuation enum: `3` for `12dB`

### Firmware build and flash

```bash
cd esp32_firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/tty.usbmodemXXXX flash monitor
```

### Notes about hardware readiness

This repo is code-ready for the ESP32 path, but some hardware-specific values still depend on your exact board:

- the actual ADC1 channel for the pin you use
- the exact ESP32-S3 board pinout
- whether you wire `LO+` and `LO-`
- signal cleanliness and grounding on the AD8232

That is normal. Those parts cannot be finalized in software alone without the board.

## 13. Raspberry Pi Deployment

Recommended deployment architecture:

- Raspberry Pi runs Mosquitto
- Raspberry Pi runs the Streamlit app
- Raspberry Pi runs ONNX inference locally
- ESP32-S3 publishes raw samples to the Pi over WiFi

### Pi setup

Install broker:

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients -y
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

Clone repo and install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export ONNX:

```bash
python export_onnx.py
```

Run app:

```bash
streamlit run app.py
```

In the app:

- `Input source` = `MQTT`
- `Inference backend` = `ONNX`
- `MQTT broker host` = `localhost`
- `MQTT topic` = `ecg/data`

In the ESP32 firmware:

- `MQTT broker URI` = `mqtt://<raspberry-pi-ip>:1883`

## 14. Current MQTT Integration Details

Python-side MQTT stream:

- implemented in [infer_stream.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/infer_stream.py:113)
- uses `paho-mqtt`
- subscribes to one topic
- converts incoming payloads into a unified sample stream

App-side MQTT wiring:

- implemented in [app.py](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/app.py:33)
- adds `MQTT` as a third input source
- allows backend choice between PyTorch and ONNX

## 15. Known Constraints and Design Choices

- The ONNX runner uses full-window inference, not hidden-state streaming.
- The PyTorch runner supports streaming-style updates using `step()`.
- The exported ONNX graph is built from the recurrent wrapper, not the FFT training graph.
- The legacy checkpoint in `models/ecg_ssm.pt` is supported.
- MQTT batching is implemented in the ESP32 firmware, but the Streamlit app still consumes a flat sample stream after parsing.

## 16. Troubleshooting

### `apt: command not found`

You are on macOS, not Raspberry Pi OS. Use `brew`, not `apt`.

### `mosquitto_pub` or `mosquitto_sub` says `Bad file descriptor`

Your Homebrew Mosquitto service probably started without a real config file. Fix with:

```bash
cp /opt/homebrew/etc/mosquitto/mosquitto.conf.example /opt/homebrew/etc/mosquitto/mosquitto.conf
brew services restart mosquitto
```

### `export_onnx.py` fails on checkpoint load

This repo already includes compatibility handling for the older checkpoint head layout.

### `export_onnx.py` fails during ONNX conversion

Make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

### Streamlit shows `Sim file not found`

You are probably still in `Simulated` mode. Switch to `MQTT` if you want live MQTT data.

### MQTT graph runs but predictions do not appear

The app waits until the rolling window reaches `187` samples before running classification.

### ESP32 publishes but the app shows no data

Check:

- broker host
- broker port
- topic name
- payload format
- whether the broker is reachable from the ESP32 network

## 17. Recommended Next Improvements

Possible future improvements:

- add timestamps to firmware batch payloads
- add an MQTT status topic for device health
- add broker authentication and TLS
- add batch-aware plotting diagnostics
- optimize ONNX runtime configuration on Raspberry Pi
- add ESP32 board-specific pin presets

## 18. Quick Start Summary

### Local laptop test

```bash
pip install -r requirements.txt
python export_onnx.py
streamlit run app.py
```

Then set `MQTT + ONNX` in the sidebar and publish test data to `ecg/data`.

### Edge deployment

```bash
# Raspberry Pi
sudo apt install mosquitto mosquitto-clients -y
pip install -r requirements.txt
python export_onnx.py
streamlit run app.py
```

```bash
# ESP32-S3
cd esp32_firmware
idf.py menuconfig
idf.py build
idf.py -p /dev/tty.usbmodemXXXX flash monitor
```

## 19. Repository Status

As of this version, the codebase supports:

- CSV simulation
- serial input
- MQTT input
- PyTorch realtime inference
- ONNX realtime inference
- ESP32-S3 MQTT publisher firmware
- batched MQTT payload support

That gives you a solid software base for plugging in the hardware later with minimal code changes.
