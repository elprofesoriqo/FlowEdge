"""Fixed-shape ONNX Runtime adapter with an explicit external-runtime boundary."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class OnnxAdapterError(RuntimeError):
    """A model/provider error with an actionable deployment path."""


class Session(Protocol):
    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...

    def run(
        self, outputs: list[str], feeds: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


class OnnxAdapter:
    """Run one fixed-shape float32 ONNX input/output pair into caller-owned output."""

    def __init__(self, session: Session):
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise OnnxAdapterError(
                "adapter requires exactly one ONNX input and output; use a model-specific boundary"
            )
        self._session, self._input, self._output = session, inputs[0], outputs[0]
        self.input_shape = self._shape(self._input, "input")
        self.output_shape = self._shape(self._output, "output")

    @classmethod
    def open(cls, model: str, providers: list[str] | None = None) -> OnnxAdapter:
        try:
            import onnxruntime as ort

            return cls(ort.InferenceSession(model, providers=providers))
        except ImportError as error:
            raise OnnxAdapterError(
                "install flowedge-onnx[runtime] to run an ONNX artifact"
            ) from error
        except Exception as error:
            raise OnnxAdapterError(
                f"ONNX Runtime could not load {model}: {error}. Use a compatible provider or convert a supported native model."
            ) from error

    @staticmethod
    def _shape(value: Any, kind: str) -> tuple[int, ...]:
        shape = getattr(value, "shape", None)
        if (
            getattr(value, "type", None) != "tensor(float)"
            or not isinstance(shape, list)
            or any(not isinstance(size, int) or size <= 0 for size in shape)
        ):
            raise OnnxAdapterError(
                f"ONNX {kind} must be fixed-shape tensor(float); dynamic shapes require a model-specific adapter"
            )
        return tuple(shape)

    def run_into(self, input: np.ndarray, output: np.ndarray) -> np.ndarray:
        self._validate(input, self.input_shape, "input")
        self._validate(output, self.output_shape, "output")
        try:
            result = self._session.run([self._output.name], {self._input.name: input})[
                0
            ]
        except Exception as error:
            raise OnnxAdapterError(
                f"ONNX Runtime execution failed: {error}. Check unsupported operators and provider support."
            ) from error
        self._validate(result, self.output_shape, "runtime output")
        np.copyto(output, result)
        return output

    @staticmethod
    def _validate(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise OnnxAdapterError(
                f"{name} must be a C-contiguous float32 array with shape {shape}"
            )
