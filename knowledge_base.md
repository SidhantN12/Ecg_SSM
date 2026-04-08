# ECG SSM Knowledge Base

This document provides a comprehensive technical reference for the `Ecg_SSM` project, detailing the architecture, theoretical underpinnings, and implementation specifics for real-time ECG classification using State Space Models (SSMs).

---

## 1. Executive Summary

The `Ecg_SSM` project implements a lightweight, dependency-free State Space Model (SSM) in PyTorch to classify ECG heartbeats in real-time. It targets the MIT-BIH Arrhythmia Database, categorizing heartbeats into five distinct classes according to AAMI standards. The system supports both simulated streaming from CSV files and live serial input from hardware sensors like the AD8232.

---

## 2. Theoretical Foundation

### 2.1 Diagonal State Space Models (DSSMs)

State Space Models are a class of sequence models that bridge the gap between Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs).

**Discrete-Time Recurrence:**
The core of the SSM is the linear recurrence relation:
$$x_{t+1} = A x_t + B u_t$$
$$y_t = C x_t + D u_t$$

Where:
- $u_t$: Input signal at time $t$.
- $x_t$: Latent state representing the "memory" of the system.
- $y_t$: Output signal.
- $A, B, C, D$: Learnable parameters.

**Diagonal Constraint:**
In this implementation, the matrix $A$ is constrained to be **diagonal**. This allows the update to be computed as an element-wise multiplication ($x_{t+1} = a \odot x_t + b \cdot u_t$), which is significantly more efficient than full matrix-vector products.

**Stability via Tanh Mapping:**
To ensure the system is stable (i.e., the state doesn't explode over long sequences), the eigenvalues of the discrete transition matrix $A$ must reside within the unit circle ($|a| < 1$). This is enforced by passing the raw parameters through a `tanh` activation function before the recurrence step.

### 2.2 ECG Signal Analysis & MIT-BIH Dataset

The project utilizes the **MIT-BIH Arrhythmia Database**, which is a standard benchmark for ECG classification.

**AAMI Heartbeat Classes:**
Heartbeats are mapped into five categories based on the Association for the Advancement of Medical Instrumentation (AAMI) standards:
1.  **N (Normal)**: Any heartbeat not classified as ectopic.
2.  **S (SVEB)**: Supraventricular ectopic beat.
3.  **V (VEB)**: Ventricular ectopic beat.
4.  **F (Fusion)**: Fusion of ventricular and normal beats.
5.  **Q (Unknown)**: Unknown or unclassifiable beat.

**Normalization (Z-Score):**
ECG signals can vary in amplitude due to electrode placement or patient physiological differences. The system employs **Z-score normalization per heartbeat**:
$$\hat{x} = \frac{x - \mu}{\sigma}$$
This ensures the model is invariant to signal scale and baseline shifts.

---

## 3. Codebase Architecture: File-by-File Analysis

### `ssm_model.py`
The core neural network definition. It implements a custom SSM layer from scratch without external libraries like `mamba-ssm`.

### `train.py`
The training pipeline for the model. Handles data downloading (via Kaggle), preprocessing, model training, and checkpointing.

### `infer_stream.py`
The inference engine. Contains logic for loading models, processing rolling windows of data, and generating streams from various sources (CSV, Serial).

### `app.py`
The Streamlit frontend. Provides a real-time dashboard for visualizing the ECG signal and model predictions.

---

## 4. Function-by-Function Reference

### `ssm_model.py`

#### `SimpleSSMLayer` (Class)
A single diagonal SSM layer.
- **`__init__(d_state, in_dim, out_dim, dropout)`**: Initializes learnable parameters `a_raw`, `b`, `c`, and `d`. `a_raw` is initialized to zero (mapping to 0 after tanh).
- **`forward(u)`**: Executes the recurrence. Takes input `u` of shape `(B, T, in_dim)`. Computes state transitions using `torch.einsum` for efficiency and returns the sequence `y` of shape `(B, T, out_dim)`.

#### `SSMEncoder` (Class)
A stack of SSM layers with non-linearities.
- **`__init__(d_state, hidden, depth, dropout)`**: Stacks multiple `SimpleSSMLayer` instances.
- **`forward(x)`**: Applies an initial linear projection, followed by a loop through SSM layers with **GELU** activation and **residual connections**. Includes LayerNorm for training stability.

#### `ECGSSMClassifier` (Class)
The top-level model for classification.
- **`forward(x)`**: Passes input through the `SSMEncoder`, followed by **Global Adaptive Average Pooling** over the time dimension and a linear head to produce logits for 5 classes.

### `train.py`

#### `ECGDataset` (Class)
Standard PyTorch Dataset wrapper for ECG data.

#### `load_kaggle_heartbeat(data_dir, auto_download)`
Locates or downloads the MIT-BIH dataset. Uses `opendatasets` to fetch from Kaggle if permitted. Returns training and testing sets as numpy arrays.

#### `normalize_per_example(X)`
Applies Z-score normalization independently to each heartbeat row in the input matrix `X`.

#### `train(args)`
The main execution loop for training. Sets up the device (CUDA/CPU), initializes the `ECGSSMClassifier`, uses `AdamW` optimizer and `CosineAnnealingLR` scheduler. Saves the final model to `ecg_ssm.pt`.

### `infer_stream.py`

#### `RealtimeRunner` (Class)
A stateful wrapper for inference during streaming.
- **`__init__(models_dir, window_size)`**: Loads the model and initializes a `deque` (walking window) to store incoming samples.
- **`step(sample)`**: The core inference step. Appends a new sample to the window. If the window is full, it normalizes the chunk, runs model inference, and returns the predicted label and class probabilities.

#### `simulate_stream_from_csv(csv_path, sample_rate_hz)`
A generator that reads heartbeats from a CSV file and yields them one-by-one at a specified frequency (default 187 Hz) to simulate a real-time sensor.

#### `serial_stream(port, baud, sample_rate_hz)`
A generator that reads from a serial port (e.g., Arduino). Expects newline-separated float values.

### `app.py`

#### `loop_stream()`
The main loop for the Streamlit app. It pulls samples from the selected generator (Simulated or Serial), updates the internal state of the `RealtimeRunner`, and refreshes the Plotly charts and class probability bars.

---

## 5. Data Specifications

- **Input Shape**: `(Batch, Time)` where `Time` is usually 187 samples (the length of one heartbeat in the MIT-BIH dataset).
- **Sampling Rate**: The dataset is sampled at **125Hz** but provided in fixed-length segments of 188 samples (last column is label). In this project, the window size defaults to 187.
- **Mapping (`LABEL_MAP`)**:
    - `0`: N (Normal)
    - `1`: S (SVEB)
    - `2`: V (VEB)
    - `3`: F (Fusion)
    - `4`: Q (Unknown)

---

## 6. Hardware Integration (Serial)

To use a live sensor:
1. Connect an **AD8232 ECG sensor** to an Arduino (e.g., Nano/Uno).
2. Upload a simple sketch that reads the analog pin and prints the value to Serial:
   ```cpp
   void setup() { Serial.begin(115200); }
   void loop() { Serial.println(analogRead(A0)); delay(4); } // ~250Hz
   ```
3. In the Streamlit sidebar, select **Input source: Serial** and specify the correct COM port.
