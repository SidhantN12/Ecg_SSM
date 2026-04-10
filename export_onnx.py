import torch
from ssm_model import ECGSSMClassifier

# Load checkpoint
ckpt = torch.load("models/ecg_ssm.pt", map_location="cpu")

# Initialize model
model = ECGSSMClassifier()

# Load ONLY the weights
model.load_state_dict(ckpt["model_state"])

model.eval()

dummy = torch.randn(1, 187)

torch.onnx.export(
    model,
    dummy,
    "models/ecg_ssm.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)
