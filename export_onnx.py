import torch
import torch.nn as nn
from ssm_model import ECGSSMClassifier, LegacyECGSSMClassifier


class ONNXExportWrapper(nn.Module):
    """
    Export-friendly wrapper that replays the encoder via recurrent step()
    instead of the FFT-based training forward path, which relies on complex ops.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, *states):
        # Flattened state input for ONNX compatibility
        # Re-bundle into what model.step_stateful expects
        h, new_states = self.model.step_stateful(x, list(states))
        # Return logits and new states
        return self.model.head(h), *new_states

# Load checkpoint
ckpt = torch.load("models/ecg_ssm.pt", map_location="cpu")

# Initialize the right model variant for this checkpoint
state = ckpt["model_state"]
if "head.0.weight" in state:
    model = ECGSSMClassifier(**ckpt["config"])
else:
    model = LegacyECGSSMClassifier(**ckpt["config"])

# Load ONLY the weights
model.load_state_dict(ckpt["model_state"])
model.eval()
export_model = ONNXExportWrapper(model).eval()

# Prepare dummy inputs for stateful step
num_layers = len(model.encoder.layers)
d_state = model.encoder.layers[0].d_state

sample = torch.randn(1, 1)
# Initial states (batch=1, d_state)
states = [torch.zeros(1, d_state) for _ in range(num_layers)]

input_names = ["input"] + [f"state_{i}" for i in range(num_layers)]
output_names = ["output"] + [f"state_{i}_out" for i in range(num_layers)]

torch.onnx.export(
    export_model,
    (sample, *states),
    "models/ecg_ssm.onnx",
    input_names=input_names,
    output_names=output_names,
    opset_version=18,
    dynamic_axes={"input": {0: "batch"}, **{f"state_{i}": {0: "batch"} for i in range(num_layers)}}
)
