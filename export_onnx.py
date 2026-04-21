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

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        states = None
        hidden_steps = []
        seq_len = x.size(1)

        for t in range(seq_len):
            xt = x[:, t].unsqueeze(1)
            h, states = self.model.step(xt, states)
            hidden_steps.append(h)

        h_all = torch.stack(hidden_steps, dim=1)
        avg_p = h_all.mean(dim=1)

        if getattr(self.model, "streaming_pool", "avgmax") == "avg":
            pooled = avg_p
        else:
            max_p = h_all.max(dim=1)[0]
            pooled = torch.cat([avg_p, max_p], dim=-1)

        return self.model.head(pooled)

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

dummy = torch.randn(1, 187)

torch.onnx.export(
    export_model,
    dummy,
    "models/ecg_ssm.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
)
