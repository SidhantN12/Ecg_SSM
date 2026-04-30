import time
import threading
from pathlib import Path
from collections import deque
from queue import Queue, Empty

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from infer_stream import mqtt_stream, ONNXRealtimeRunner

st.set_page_config(page_title="ECG Architecture v1", layout="wide")
st.title("ECG Real-time Visualization & Inference (Architecture v1)")

# ---------------------------------------------------------------------------
# Shared state between background thread and Streamlit UI
# ---------------------------------------------------------------------------
# We use a single dict cached at the resource level so the background thread
# and the Streamlit script share the exact same object across reruns.

@st.cache_resource
def get_shared():
    return {
        "running": False,
        "signal_buf": deque(maxlen=187 * 20),   # up to 20 s at 187 Hz
        "time_buf":   deque(maxlen=187 * 20),
        "label": None,
        "probs": None,
        "error": None,
        "thread": None,
    }

shared = get_shared()

# ---------------------------------------------------------------------------
# HTTP server for Android companion app
# ---------------------------------------------------------------------------

@st.cache_resource
def get_prediction_state():
    return {"label": "Waiting...", "confidence": 0.0}

prediction_state = get_prediction_state()

_fastapi_app = FastAPI()
_fastapi_app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@_fastapi_app.get("/latest")
def get_latest():
    return prediction_state

@st.cache_resource
def start_http_server():
    t = threading.Thread(
        target=lambda: uvicorn.run(_fastapi_app, host="0.0.0.0", port=8000, log_level="error"),
        daemon=True,
    )
    t.start()
    return t

start_http_server()

# ---------------------------------------------------------------------------
# Background worker — runs entirely outside Streamlit's script context
# ---------------------------------------------------------------------------

def _inference_worker(host: str, port: int, topic: str, sample_rate: int, display_seconds: int):
    """Runs in a daemon thread. Reads MQTT, updates shared dict."""
    try:
        runner = ONNXRealtimeRunner(models_dir=Path("models"))
    except Exception as e:
        shared["error"] = str(e)
        shared["running"] = False
        return

    max_points = sample_rate * display_seconds
    # Replace the deques with fresh ones of the right size
    shared["signal_buf"] = deque(maxlen=max_points)
    shared["time_buf"]   = deque(maxlen=max_points)

    start_t = time.time()

    try:
        for s in mqtt_stream(host=host, port=port, topic=topic):
            if not shared["running"]:
                break

            now = time.time() - start_t
            shared["signal_buf"].append(float(s))
            shared["time_buf"].append(now)

            result = runner.step(float(s))
            if result:
                label, probs = result
                shared["label"] = label
                shared["probs"] = probs
                prediction_state["label"] = label
                prediction_state["confidence"] = max(probs.values())

    except Exception as e:
        shared["error"] = str(e)
    finally:
        shared["running"] = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("MQTT Settings")
    mqtt_host  = st.text_input("Broker Host", value="localhost")
    mqtt_port  = st.number_input("Broker Port", min_value=1, max_value=65535, value=1883, step=1)
    mqtt_topic = st.text_input("Topic", value="ecg/data")

    st.header("Processing Settings")
    SAMPLE_RATE     = 187
    display_seconds = st.slider("Display Window (sec)", 1, 20, 5)

    onnx_path = Path("models") / "ecg_ssm.onnx"
    if not onnx_path.exists():
        st.warning("⚠️ `models/ecg_ssm.onnx` not found. Run `python export_onnx.py` first.")

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("Start")
    with col2:
        stop_clicked  = st.button("Stop")

if start_clicked and not shared["running"]:
    shared["running"] = True
    shared["error"]   = None
    t = threading.Thread(
        target=_inference_worker,
        args=(mqtt_host, int(mqtt_port), mqtt_topic, SAMPLE_RATE, display_seconds),
        daemon=True,
    )
    t.start()
    shared["thread"] = t

if stop_clicked:
    shared["running"] = False

# ---------------------------------------------------------------------------
# UI — render current snapshot from shared state
# ---------------------------------------------------------------------------

placeholder_plot  = st.empty()
c1, c2, c3 = st.columns(3)
with c1:  placeholder_rate  = st.empty()
with c2:  placeholder_label = st.empty()
with c3:  placeholder_prob  = st.empty()

if shared["error"]:
    st.error(f"Error: {shared['error']}")

if shared["running"] or shared["label"] is not None:
    # Plot
    xs = np.array(shared["time_buf"])
    ys = np.array(shared["signal_buf"])
    if xs.size:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name="Filtered ECG",
            line=dict(color="#00FF00", width=2),
        ))
        fig.update_layout(
            height=400, template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Time (s)", showgrid=False),
            yaxis=dict(title="Amplitude",  showgrid=True),
        )
        placeholder_plot.plotly_chart(fig, use_container_width=True)

    placeholder_rate.metric("Live Sample Rate", f"{SAMPLE_RATE} Hz")

    if shared["label"]:
        placeholder_label.metric("Diagnosis", shared["label"])
        max_p = max(shared["probs"].values())
        placeholder_prob.progress(max_p, text=f"Confidence: {max_p:.1%}")

    # Keep re-running the script at ~20 Hz while the worker is active
    if shared["running"]:
        time.sleep(0.05)
        st.rerun()

else:
    st.info("Configure MQTT settings and press 'Start' to begin visualization.")