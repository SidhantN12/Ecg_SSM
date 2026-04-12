# ESP32-S3 MQTT Publisher

This firmware publishes raw AD8232 ECG samples from an ESP32-S3 to the MQTT topic used by the Python app:

- Topic: `ecg/data`
- Payload: plain integer sample or comma-separated sample batch
- Transport: WiFi + MQTT

## Configure

From `esp32_firmware/` run:

```bash
idf.py menuconfig
```

Set values under `ECG MQTT Publisher`:

- `WiFi SSID`
- `WiFi password`
- `MQTT broker URI`
- `MQTT publish topic`
- `MQTT samples per publish`
- `Sampling rate (Hz)`
- `ADC1 channel for AD8232 output`
- optional `LO+` / `LO-` GPIOs

Recommended defaults:

- MQTT topic: `ecg/data`
- MQTT QoS: `0` for lowest latency
- MQTT samples per publish: `8`
- Sampling rate: `250`
- ADC attenuation enum: `3` (`12dB`)

## Build and flash

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/tty.usbmodemXXXX flash monitor
```

## Streamlit side

In the repo root app:

- `Input source` = `MQTT`
- `Inference backend` = `ONNX`
- `MQTT broker host` = the machine running Mosquitto
- `MQTT topic` = `ecg/data`

## Notes

- This firmware publishes raw ADC samples only. Inference stays on the Pi or laptop.
- The Python MQTT subscriber accepts single values, comma-separated batches, and JSON `samples` arrays.
- `LO+` and `LO-` can be left disabled by setting them to `-1`.
- Real ADC channel selection depends on your exact ESP32-S3 board pinout.
