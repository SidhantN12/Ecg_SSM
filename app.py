import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from infer_stream import mqtt_stream


st.set_page_config(page_title="ECG Architecture v1", layout="wide")
st.title("ECG Real-time Visualization (Architecture v1)")

with st.sidebar:
    st.header("MQTT Settings")
    mqtt_host = st.text_input("Broker Host", value="localhost")
    mqtt_port = st.number_input("Broker Port", min_value=1, max_value=65535, value=1883, step=1)
    mqtt_topic = st.text_input("Topic", value="ecg/data")
    sample_rate = st.number_input("Sample Rate (Hz)", min_value=50, max_value=1000, value=250, step=1)
    display_seconds = st.slider("Display Window (sec)", 1, 20, 5)
    run_btn = st.button("Start / Restart Stream")


if "samples" not in st.session_state or run_btn:
    st.session_state.samples = []
    st.session_state.timestamps = []
    st.session_state.start_time = time.time()


placeholder_plot = st.empty()
placeholder_metrics = st.empty()


def render_plot(xs, ys):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, 
        y=ys, 
        mode='lines', 
        name='Filtered ECG',
        line=dict(color='#00FF00', width=2)
    ))
    fig.update_layout(
        height=400, 
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Time (s)", showgrid=False),
        yaxis=dict(title="Amplitude", showgrid=True),
    )
    placeholder_plot.plotly_chart(fig, use_container_width=True)


def loop_stream():
    gen = mqtt_stream(host=mqtt_host, port=int(mqtt_port), topic=mqtt_topic)
    start = st.session_state.start_time
    
    last_ui_update = 0
    ui_update_interval = 1.0 / 25.0 # 25Hz UI refresh
    
    try:
        for s in gen:
            now = time.time() - start
            st.session_state.samples.append(float(s))
            st.session_state.timestamps.append(now)
            
            # Maintenance: Keep display window
            max_points = int(sample_rate * display_seconds)
            if len(st.session_state.samples) > max_points:
                st.session_state.samples = st.session_state.samples[-max_points:]
                st.session_state.timestamps = st.session_state.timestamps[-max_points:]
            
            # UI Throttling
            current_time = time.time()
            if current_time - last_ui_update > ui_update_interval:
                render_plot(st.session_state.timestamps, st.session_state.samples)
                placeholder_metrics.metric("Live Sample Rate", f"{sample_rate} Hz", delta=None)
                last_ui_update = current_time
                
            # Allow Streamlit context to breathe
            time.sleep(0.001)
    except Exception as e:
        st.error(f"Stream Error: {e}")


if run_btn:
    loop_stream()
else:
    st.info("Configure MQTT settings and press 'Start' to begin visualization.")
