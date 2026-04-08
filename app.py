import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from infer_stream import RealtimeRunner, simulate_stream_from_csv, serial_stream


st.set_page_config(page_title="ECG SSM Realtime", layout="wide")
st.title("ECG Realtime Classification (State-Space Model)")

def find_default_sim_file(base_dir: Path) -> str:
    base = Path(base_dir)
    # Try common locations
    candidates = [
        base / "heartbeat" / "mitbih_test.csv",
        base / "mitbih_test.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fallback: search
    hits = list(base.glob("**/mitbih_test.csv"))
    if hits:
        return str(hits[0])
    return str(candidates[0])


with st.sidebar:
    st.header("Settings")
    models_dir = st.text_input("Models dir", value="models")
    data_dir = st.text_input("Data dir (for simulation)", value="data")
    source = st.selectbox("Input source", ["Simulated", "Serial"], index=0)
    window_size = st.number_input("Window size (samples)", min_value=64, max_value=1024, value=187, step=1)
    sample_rate = st.number_input("Sample rate (Hz)", min_value=50, max_value=1000, value=187, step=1)
    if source == "Serial":
        com_port = st.text_input("COM port (e.g., COM3)", value="COM3")
        baud = st.number_input("Baud rate", min_value=9600, max_value=921600, value=115200, step=1)
        sim_file = ""
    else:
        default_sim = find_default_sim_file(Path(data_dir))
        sim_file = st.text_input("Sim file (mitbih_test.csv)", value=default_sim)
    run_btn = st.button("Run / Refresh")


def model_ready(models_dir: str) -> bool:
    return Path(models_dir).joinpath("ecg_ssm.pt").exists()


if "runner" not in st.session_state or run_btn:
    if not model_ready(models_dir):
        st.error(f"Model not found at {Path(models_dir) / 'ecg_ssm.pt'}. Train first (python train.py).")
    else:
        try:
            st.session_state.runner = RealtimeRunner(models_dir=Path(models_dir), window_size=int(window_size))
            st.session_state.last_probs = None
            st.session_state.label = "—"
            st.session_state.samples = []
            st.session_state.timestamps = []
            st.session_state.start_time = time.time()
        except Exception as e:
            st.error(f"Failed to init model: {e}")


placeholder_plot = st.empty()
placeholder_label = st.empty()
placeholder_probs = st.empty()


def render_plot(xs, ys):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='ECG'))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    placeholder_plot.plotly_chart(fig, use_container_width=True)


def render_probs(probs_dict):
    if probs_dict is None:
        placeholder_probs.write("Waiting for window...")
        return
    df = pd.DataFrame({"Class": list(probs_dict.keys()), "Prob": list(probs_dict.values())})
    df.sort_values("Prob", ascending=False, inplace=True)
    placeholder_probs.bar_chart(df.set_index("Class"))


def loop_stream():
    # Choose generator
    if source == "Serial":
        gen = serial_stream(port=com_port, baud=int(baud), sample_rate_hz=int(sample_rate))
    else:
        csv_path = Path(sim_file)
        if not csv_path.exists():
            st.warning(f"Sim file not found: {csv_path}. Update 'Data dir' or 'Sim file' in the sidebar.")
            return
        gen = simulate_stream_from_csv(csv_path, sample_rate_hz=int(sample_rate), loop=True)

    runner = st.session_state.runner
    start = st.session_state.start_time
    
    # UI Throttling
    last_ui_update = 0
    ui_update_interval = 1.0 / 30.0 # 30Hz cap
    
    try:
        while True:
            s = next(gen)
            now = time.time() - start
            st.session_state.samples.append(float(s))
            st.session_state.timestamps.append(now)
            
            # Keep last few seconds visible
            max_points = int(sample_rate * 5) # Show 5 seconds instead of 10 for speed
            if len(st.session_state.samples) > max_points:
                st.session_state.samples = st.session_state.samples[-max_points:]
                st.session_state.timestamps = st.session_state.timestamps[-max_points:]

            out = runner.step(float(s))
            if out is not None:
                label, probs = out
                st.session_state.label = label
                st.session_state.last_probs = probs

            # Throttle UI updates to 30Hz
            current_time = time.time()
            if current_time - last_ui_update > ui_update_interval:
                render_plot(st.session_state.timestamps, st.session_state.samples)
                placeholder_label.markdown(f"### Prediction: {st.session_state.label}")
                render_probs(st.session_state.last_probs)
                last_ui_update = current_time

            # Small sleep to allow background tasks/UI to keep up
            time.sleep(0.001)
    except StopIteration:
        pass


try:
    loop_stream()
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
