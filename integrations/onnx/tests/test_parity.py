import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from flowedge_onnx import OnnxAdapter


class OnnxParityTests(unittest.TestCase):
    def test_exported_fixed_policy_matches_onnx_runtime(self):
        import onnxruntime as ort
        import torch

        class FixedPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.tensor([[0.25, -0.5, 0.75], [-1.0, 0.5, 0.125]])
                )
                self.register_buffer("bias", torch.tensor([0.1, -0.2]))

            def forward(self, condition):
                return torch.tanh(condition @ self.weight.T + self.bias)

        condition = torch.tensor([[0.5, -0.25, 1.0]], dtype=torch.float32)
        policy = FixedPolicy().eval()
        expected = policy(condition).detach().numpy()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fixed_policy.onnx"
            with redirect_stdout(StringIO()):
                torch.onnx.export(
                    policy,
                    condition,
                    model,
                    input_names=["condition"],
                    output_names=["action"],
                    opset_version=18,
                )
            output = np.empty_like(expected)
            OnnxAdapter(ort.InferenceSession(model)).run_into(condition.numpy(), output)
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
