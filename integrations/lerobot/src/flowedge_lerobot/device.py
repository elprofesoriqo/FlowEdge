"""Fail-closed CPU/CUDA selection for the LeRobot adapter."""


def config_device(config):
    return str(getattr(config, "device", "cpu") or "cpu")


def require_native_device(device, flowedge_module, engine=None):
    if device in (None, "cpu"):
        return
    if device != "cuda" and not str(device).startswith("cuda"):
        raise ValueError("device must be cpu or cuda")
    if not getattr(flowedge_module, "cuda", False):
        raise RuntimeError(
            "CUDA requested but FlowEdge was built without FLOWEDGE_BACKEND=cuda. "
            "v0.1.1 wheels are CPU and are not on PyPI; CUDA is a source build with nvcc."
        )
    if engine is not None and not getattr(engine, "cuda_resident", False):
        raise RuntimeError(
            "CUDA requested but this engine is running CPU kernels "
            "(device-resident load failed, 4 GB VRAM OOM, no GPU, or "
            "FLOWEDGE_CUDA_FORCE_HOST=1). Use device=cpu, or set "
            "FLOWEDGE_CUDA_REQUIRED=1 to fail at load."
        )
