import time
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from infer_stream import mqtt_stream, RealtimeRunner


st.set_page_config(page_title="ECG Architecture v1", layout="wide")
st.title("ECG Real-time Visualization & Inference (Architecture v1)")

# --- Session State Initialization ---
if "running" not in st.session_state:
    st.session_state.running = False

with st.sidebar:
    st.header("MQTT Settings")
    mqtt_host = st.text_input("Broker Host", value="localhost")
    mqtt_port = st.number_input("Broker Port", min_value=1, max_value=65535, value=1883, step=1)
    mqtt_topic = st.text_input("Topic", value="ecg/data")
    
    st.header("Processing Settings")
    sample_rate = st.number_input("Sample Rate (Hz)", min_value=50, max_value=1000, value=250, step=1)
    norm_window = st.slider("Normalization Window (Samples)", 50, 500, 187)
    display_seconds = st.slider("Display Window (sec)", 1, 20, 5)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start"):
            st.session_state.running = True
    with col2:
        if st.button("Stop"):
            st.session_state.running = False

# --- UI Containers ---
placeholder_plot = st.empty()
c1, c2, c3 = st.columns(3)
with c1:
    placeholder_rate = st.empty()
with c2:
    placeholder_label = st.empty()
with c3:
    placeholder_prob = st.empty()

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

def main_loop():
    # Initialize Runner
    runner = RealtimeRunner(models_dir=Path("models"), window_size=norm_window)
    
    # Initialize Deques for plotting (Circular buffers)
    max_points = int(sample_rate * display_seconds)
    y_buffer = deque(maxlen=max_points)
    x_buffer = deque(maxlen=max_points)
    
    gen = mqtt_stream(host=mqtt_host, port=int(mqtt_port), topic=mqtt_topic)
    start_t = time.time()
    
    last_ui_update = 0
    ui_update_interval = 1.0 / 20.0 # 20Hz UI refresh
    
    last_inference_t = 0
    inference_interval = 1.0 / 10.0 # 10Hz Inference check
    
    try:
        for s in gen:
            if not st.session_state.running:
                break
                
            now = time.time() - start_t
            y_buffer.append(float(s))
            x_buffer.append(now)
            
            # Inference Step
            inf_res = runner.step(float(s))
            
            # UI Throttling
            current_time = time.time()
            if current_time - last_ui_update > ui_update_interval:
                render_plot(np.array(x_buffer), np.array(y_buffer))
                placeholder_rate.metric("Live Sample Rate", f"{sample_rate} Hz")
                
                if inf_res:
                    label, probs = inf_res
                    placeholder_label.metric("Diagnosis", label)
                    max_p = max(probs.values())
                    placeholder_prob.progress(max_p, text=f"Confidence: {max_p:.1%}")
                
                last_ui_update = current_time
            
            # Allow Streamlit to remain responsive
            time.sleep(0.0001)
            
    except Exception as e:
        st.error(f"Stream Error: {e}")
        st.session_state.running = False


if st.session_state.running:
    main_loop()
else:
    st.info("Configure MQTT settings and press 'Start' to begin visualization.")
