from pathlib import Path
import numpy as np
import onnxruntime as ort

LABEL_MAP = {
    0: "N (Normal)",
    1: "S (SVEB)",
    2: "V (VEB)",
    3: "F (Fusion)",
    4: "Q (Unknown)",
}

model_path = Path("models/ecg_ssm.onnx")

sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

print("Inputs:")
for i in sess.get_inputs():
    print(" ", i.name, i.shape, i.type)

print("Outputs:")
for o in sess.get_outputs():
    print(" ", o.name, o.shape, o.type)

feeds = {}

for inp in sess.get_inputs():
    # Replace dynamic dims with 1 for a smoke test
    shape = []
    for d in inp.shape:
        if isinstance(d, int) and d > 0:
            shape.append(d)
        else:
            shape.append(1)

    if inp.name == "input":
        # Most likely a full ECG window
        if len(shape) == 2 and shape[1] == 187:
            x = np.random.randn(1, 187).astype(np.float32)
        elif len(shape) == 3 and shape[1] == 187 and shape[2] == 1:
            x = np.random.randn(1, 187, 1).astype(np.float32)
        else:
            x = np.random.randn(*shape).astype(np.float32)
        feeds[inp.name] = x
    else:
        # State tensors, if your export included them
        feeds[inp.name] = np.zeros(shape, dtype=np.float32)

outputs = sess.run(None, feeds)

print("\nRaw outputs:")
for idx, out in enumerate(outputs):
    print(f"output[{idx}] shape={np.array(out).shape}")

logits = np.array(outputs[0])
if logits.ndim == 2:
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = int(np.argmax(probs[0]))
    print("\nPrediction:", LABEL_MAP.get(pred, str(pred)))
    print("Probabilities:", probs[0])