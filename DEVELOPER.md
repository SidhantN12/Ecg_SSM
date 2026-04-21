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
*   **Sample Rate**: The firmware dynamically calculates biquad coefficients to match this. Default: 250 Hz.

### 4.3 Build and Flash
```bash
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
4.  In the sidebar, enter the RPi's IP address (or `localhost` if running locally) and click **Start Stream**.

---

## 6. Future Scope: Architecture v2

The Architecture v1 refactor has modularized the acquisition and signal processing layers to simplify the **Architecture v2** migration:

### 6.1 Modular Acquisition (RP2040)
*   The ESP32's `adc_continuous` driver will be replaced by a dedicated **RP2040** module.
*   Because acquisition is decoupled from transport in `main.cpp`, the ESP32 will simply swap the DMA reader task for an SPI/UART listener task while the rest of the telemetry pipeline remains unchanged.

### 6.2 Hardware Filtering (SLG47910V)
*   The `EcgFilterPipeline` class in `filter_pipeline.hpp` serves as a software wrapper.
*   In v2, this wrapper will be updated to interface with the **SLG47910V GreenPAK FPGA** via I2C/SPI, allowing the CPU to bypass software biquads entirely while maintaining the same `apply()` interface for the system.

### 6.3 System Orchestration (ESP32)
*   The ESP32's role transitions to a **System Orchestrator**, managing the lifecycle of edge components and handling secure WiFi/BLE communication.
