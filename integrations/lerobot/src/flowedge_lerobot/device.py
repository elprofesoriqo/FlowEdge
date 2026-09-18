"""Fail-closed CPU/CUDA selection for the LeRobot adapter."""


def config_device(config):
    return str(getattr(config, "device", "cpu") or "cpu")


def require_native_device(device, flowedge_module):
    if device in (None, "cpu"):
        return
    if device != "cuda" and not str(device).startswith("cuda"):
        raise ValueError("device must be cpu or cuda")
    if not getattr(flowedge_module, "cuda", False):
        raise RuntimeError(
            "CUDA requested but FlowEdge was built without FLOWEDGE_BACKEND=cuda"
        )
