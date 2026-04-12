# Quickstart

This is the fast path for getting the ECG SSM project running.

For the full architecture and detailed setup, see [README.md](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/README.md:1).

## 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Start MQTT broker

### macOS

```bash
brew install mosquitto
cp /opt/homebrew/etc/mosquitto/mosquitto.conf.example /opt/homebrew/etc/mosquitto/mosquitto.conf
brew services start mosquitto
```

### Raspberry Pi

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients -y
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

## 3. Export ONNX model

```bash
python export_onnx.py
```

Expected file:

```bash
models/ecg_ssm.onnx
```

## 4. Run the dashboard

```bash
streamlit run app.py
```

In the sidebar select:

- `Input source` = `MQTT`
- `Inference backend` = `ONNX`
- `MQTT broker host` = `localhost`
- `MQTT broker port` = `1883`
- `MQTT topic` = `ecg/data`

## 5. Publish fake ECG samples

### Single-sample mode

```bash
while true; do mosquitto_pub -h localhost -t ecg/data -m "$((RANDOM % 200 + 400))"; sleep 0.01; done
```

### Batch mode

```bash
while true; do mosquitto_pub -h localhost -t ecg/data -m "500,502,498,503,507,509,506,504"; sleep 0.05; done
```

The app will begin predicting once it has collected `187` samples.

## 6. MQTT payloads supported

The app accepts:

```text
512
512,515,498,502
```

```json
[512, 515, 498, 502]
```

```json
{"samples": [512, 515, 498, 502]}
```

## 7. ESP32-S3 firmware quick path

Firmware lives in [esp32_firmware/](/Users/srivaishnavikodali/Desktop/sem_6/Ecg_SSM-1/esp32_firmware/README.md:1).

When you have hardware:

```bash
cd esp32_firmware
idf.py menuconfig
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/tty.usbmodemXXXX flash monitor
```

Set in `menuconfig`:

- WiFi SSID/password
- MQTT broker URI
- MQTT topic = `ecg/data`
- MQTT samples per publish = `8`
- ADC1 channel for AD8232 output

## 8. Raspberry Pi deployment quick path

On the Pi:

```bash
sudo apt install mosquitto mosquitto-clients -y
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python export_onnx.py
streamlit run app.py
```

Then set:

- `Input source` = `MQTT`
- `Inference backend` = `ONNX`
- `MQTT broker host` = `localhost`

In ESP32 firmware:

- `MQTT broker URI` = `mqtt://<raspberry-pi-ip>:1883`

## 9. Common issues

### `apt: command not found`

You are on macOS. Use `brew`.

### `Bad file descriptor` from Mosquitto tools

Create the config file and restart Mosquitto:

```bash
cp /opt/homebrew/etc/mosquitto/mosquitto.conf.example /opt/homebrew/etc/mosquitto/mosquitto.conf
brew services restart mosquitto
```

### No predictions yet

The model waits for a full `187`-sample window.
