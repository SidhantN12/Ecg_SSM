# Quickstart: Architecture v1

This is the fast path for getting the ECG SSM project running.

For the full architecture and detailed setup, see [README.md](README.md).

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Start MQTT Broker

### Raspberry Pi / Linux

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients -y
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

## 3. Run the Dashboard

```bash
streamlit run app.py
```

In the sidebar:
- Enter your **MQTT Broker Host**.
- Click **Start**.

> **Note:** Sample rate (187 Hz) and normalization window (187 samples) are fixed system-wide and cannot be changed in the sidebar.

## 4. ESP32 Firmware (Architecture v1)

Firmware lives in [esp32_firmware/](esp32_firmware/README.md).

When you have hardware (AD8232 + ESP32 WROOM 32):

```bash
cd esp32_firmware
idf.py set-target esp32
idf.py menuconfig
idf.py build
idf.py -p <YOUR_PORT> flash monitor
```

Set in `menuconfig` under **ECG Project Configuration**:
- WiFi SSID/password.
- MQTT broker URI (`mqtt://<BROKER_IP>:1883`).
- Batch size (e.g., 8).

> **Note:** Sample rate is hardcoded to **187 Hz** in firmware — do not add it to menuconfig.

## 5. Telemetry Protocol

This project uses a high-performance **Binary Telemetry Protocol** to ensure zero-jitter and high integrity.

**Packet Structure**:
- Header: `0xA5` (1 byte)
- Sequence: `uint32_t` (4 bytes)
- Count: `uint8_t` (1 byte)
- Data: `count * float32` (Little Endian)
- Integrity: `CRC8` (1 byte)

## 6. Common Issues

### No predictions yet
The model waits for a **warm-up window of 187 samples** before generating the first diagnosis. This is a fixed system constant and is not adjustable.

### Dashboard shows a flatline
Electrodes are likely disconnected. The firmware's lead-off detection (`LO+`/`LO-`) outputs `0.0` when leads are off. Re-attach electrodes to resume signal.

### `WARNING: Packet Loss Detected` in terminal
MQTT packets arrived out of sequence. Check your WiFi signal and broker load. The stream will continue; lost packets are logged but not recovered.

### Portability to Architecture v2
Because the Python inference engine is decoupled from the sampling source, migrating to the v2 hardware (RP2040 sampler) only requires updating the `acquisition_task` on the ESP32 while keeping the dashboard and model untouched.
