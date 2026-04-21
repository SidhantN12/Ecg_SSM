# Architecture v1: ESP32 MQTT Publisher

This firmware publishes filtered AD8232 ECG samples from an ESP32 WROOM 32 to the MQTT topic used by the Python app.

- **Sample Rate**: Configurable (Default 250Hz).
- **Transport**: DMA-based ADC continuous sampling.
- **Protocol**: High-performance **Binary Protocol** with Header, Sequence, and CRC8 verification.
- **Connectivity**: WiFi + MQTT.

## Configure

From `esp32_firmware/` run:

```bash
idf.py menuconfig
```

Set values under `ECG Project Configuration`:

- `WiFi SSID` & `password`
- `MQTT broker URI` (e.g. `mqtt://192.168.1.50`)
- `MQTT samples per publish` (Batch Size)
- `Sampling rate (Hz)`

## Build and flash

```bash
idf.py set-target esp32
idf.py build
idf.py -p <YOUR_PORT> flash monitor
```

## Streamlit side

In the repo root app:

- `Broker Host` = The machine running the MQTT broker (Mosquitto).
- `Topic` = `ecg/data`.
- `Normalization Window` = Adjust for real-world signal drift.

## Notes

- **High Fidelity**: Architecture v1 uses the ESP32's DMA controller to ensure zero-jitter sampling, which is critical for the software biquad filters.
- **Architecture v2**: This firmware is designed as the "Orchestrator" for v2. In v2, the `acquisition_task` will be updated to read from the RP2040/FPGA module instead of the internal ADC.
