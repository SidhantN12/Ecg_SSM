<<<<<<< HEAD
# Ecg_SSM
=======
ECG State-Space Model (SSM) — Realtime Demo

What you get
- Simple, self-contained PyTorch SSM (no exotic deps)
- Quick trainer on Kaggle ECG Heartbeat dataset
- Streamlit app for realtime visualization and predictions
- Supports simulated stream (from dataset) or serial input (e.g., AD8232 + Arduino)

Quick start
1) Create venv and install deps
   - Windows PowerShell
     python -m venv .venv
     .venv\\Scripts\\Activate.ps1
     pip install -r requirements.txt

2) Download data + train (small demo)
   - You need a Kaggle account and API token (kaggle.json). See below.
     python train.py --epochs 5

3) Run realtime app (simulated by default)
   streamlit run app.py

Kaggle setup (once)
- Make a Kaggle account → Account → Create New API Token → downloads kaggle.json
- Place kaggle.json in one of:
  - %USERPROFILE%\.kaggle\kaggle.json (Windows)
  - Or set env vars: KAGGLE_USERNAME, KAGGLE_KEY

Serial input (optional)
- Hardware: simple ECG sensor (e.g., AD8232) + Arduino/Nano/ESP32, send integer samples via Serial at a known sample rate.
- App settings: choose "Serial", set COM port (e.g., COM3) and sample rate (e.g., 250 Hz).

Notes
- Training uses the Kaggle ECG Heartbeat (MIT-BIH subset). Labels: {0:N,1:S,2:V,3:F,4:Q}.
- The SSM here is a simple diagonal state-space layer (learnable, stable) stacked a few times for a lightweight, dependency-free SSM.
- You can increase epochs and model width for accuracy.

>>>>>>> 7d6ce75 (Initial commit: ECG SSM realtime project)
