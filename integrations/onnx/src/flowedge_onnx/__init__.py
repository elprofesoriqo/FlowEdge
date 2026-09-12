"""Optional ONNX Runtime deployment boundary; never imported by FlowEdge Core."""

from .adapter import OnnxAdapter, OnnxAdapterError

__all__ = ["OnnxAdapter", "OnnxAdapterError"]
