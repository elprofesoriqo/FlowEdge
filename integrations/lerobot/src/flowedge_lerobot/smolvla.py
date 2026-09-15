"""Explicit LeRobot cache boundary for FlowEdge's native SmolVLA action expert.

This module deliberately does not run observation preprocessing, image/language
encoding, or the visual-language model. A source-compatible caller supplies
the VLM cache once per action chunk; FlowEdge executes the seeded Euler action
expert and this adapter exposes the robot's unpadded action coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


SMOLVLA_ACTION_EXPERT_ARCHITECTURE = 7
SMOLVLA_KEY_VALUE_WIDTH = 320


class SmolVLAEngine(Protocol):
    """Native engine operations consumed by the cache-bound adapter."""

    @property
    def model_metadata(self) -> dict[str, Any]: ...

    def smolvla_sample(
        self,
        initial_noise: np.ndarray,
        prefix_keys: np.ndarray,
        prefix_values: np.ndarray,
        prefix_mask: np.ndarray,
        steps: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class SmolVLAActionContract:
    """Fixed action-expert dimensions reported by a pinned native checkpoint."""

    max_action_dim: int
    chunk_size: int
    expert_layers: int
    key_value_width: int = SMOLVLA_KEY_VALUE_WIDTH

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> SmolVLAActionContract:
        required = ("architecture", "action_dim", "action_horizon", "n_layers")
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError(f"missing SmolVLA model metadata: {', '.join(missing)}")
        if int(metadata["architecture"]) != SMOLVLA_ACTION_EXPERT_ARCHITECTURE:
            raise ValueError("engine is not a FlowEdge SmolVLA action-expert checkpoint")
        contract = cls(
            max_action_dim=int(metadata["action_dim"]),
            chunk_size=int(metadata["action_horizon"]),
            expert_layers=int(metadata["n_layers"]),
        )
        if min(contract.max_action_dim, contract.chunk_size, contract.expert_layers) <= 0:
            raise ValueError("SmolVLA action-expert metadata must contain positive dimensions")
        return contract


@dataclass(frozen=True)
class SmolVLAKVCache:
    """A batch-one, layer-major VLM K/V cache from a source-compatible encoder."""

    keys: np.ndarray
    values: np.ndarray
    mask: np.ndarray

    def validate(self, contract: SmolVLAActionContract) -> None:
        prefix_length = self.mask.size
        expected_cache_shape = (
            contract.expert_layers,
            prefix_length,
            contract.key_value_width,
        )
        for name, value in (("keys", self.keys), ("values", self.values)):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or not value.flags.c_contiguous
                or value.shape != expected_cache_shape
            ):
                raise ValueError(
                    f"{name} must be a C-contiguous float32 array with shape {expected_cache_shape}"
                )
        if (
            not isinstance(self.mask, np.ndarray)
            or self.mask.dtype != np.uint8
            or not self.mask.flags.c_contiguous
            or self.mask.ndim != 1
            or not 0 < prefix_length <= 512
            or self.mask[0] != 1
            or not np.isin(self.mask, (0, 1)).all()
            or np.any(self.mask[1:] > self.mask[:-1])
        ):
            raise ValueError("mask must be 1..512 C-contiguous uint8 leading ones then trailing zeros")


class FlowEdgeSmolVLACachedExpert:
    """Seeded action-chunk executor over a caller-supplied SmolVLA VLM cache.

    One instance has reusable noise and action workspaces and is therefore
    single-threaded. The returned values are only the first ``action_dim``
    coordinates of the padded SmolVLA action representation; callers apply
    the same LeRobot postprocessing used by their source policy.
    """

    def __init__(
        self,
        engine: SmolVLAEngine,
        *,
        action_dim: int,
        action_steps: int | None = None,
        seed: int = 0,
    ):
        self.contract = SmolVLAActionContract.from_metadata(engine.model_metadata)
        if action_dim <= 0 or action_dim > self.contract.max_action_dim:
            raise ValueError("action_dim must be within the checkpoint's padded action dimension")
        if action_steps is None:
            action_steps = self.contract.chunk_size
        if action_steps <= 0 or action_steps > self.contract.chunk_size:
            raise ValueError("action_steps must be within the checkpoint action chunk")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.action_dim = action_dim
        self.action_steps = action_steps
        self._engine = engine
        self._seed = seed
        self._noise = np.empty(
            (self.contract.chunk_size, self.contract.max_action_dim), dtype=np.float32
        )
        self._padded_actions = np.empty_like(self._noise)
        self._chunk = np.empty((self.action_steps, self.action_dim), dtype=np.float32)
        self.reset()

    @property
    def remaining_actions(self) -> int:
        """Number of sampled actions still available through ``take_action_into``."""
        return self.action_steps - self._next_action

    def reset(self) -> None:
        """Discard a stale action chunk and reset seeded sampling for a new episode."""
        self._rng = np.random.default_rng(self._seed)
        self._next_action = self.action_steps

    @staticmethod
    def _validate_output(output: np.ndarray, shape: tuple[int, ...]) -> None:
        if (
            not isinstance(output, np.ndarray)
            or output.dtype != np.float32
            or not output.flags.c_contiguous
            or output.shape != shape
        ):
            raise ValueError(f"output must be a C-contiguous float32 array with shape {shape}")

    def _prepare_noise(self, noise: np.ndarray | None) -> None:
        if noise is None:
            self._rng.standard_normal(self._noise.shape, dtype=np.float32, out=self._noise)
            return
        candidate = np.asarray(noise)
        if (
            candidate.dtype != np.float32
            or candidate.shape != self._noise.shape
            or not candidate.flags.c_contiguous
            or not np.isfinite(candidate).all()
        ):
            raise ValueError(
                "noise must be finite C-contiguous float32 with the full padded action shape"
            )
        np.copyto(self._noise, candidate)

    def _sample_into(
        self,
        cache: SmolVLAKVCache,
        output: np.ndarray,
        *,
        noise: np.ndarray | None,
        steps: int,
    ) -> None:
        if steps <= 0 or steps > 100:
            raise ValueError("steps must be in [1, 100]")
        cache.validate(self.contract)
        self._prepare_noise(noise)
        sampled = np.asarray(
            self._engine.smolvla_sample(
                self._noise, cache.keys, cache.values, cache.mask, steps=steps
            )
        )
        if (
            sampled.dtype != np.float32
            or sampled.shape != self._padded_actions.shape
            or not np.isfinite(sampled).all()
        ):
            raise RuntimeError("FlowEdge returned an invalid padded SmolVLA action chunk")
        np.copyto(self._padded_actions, sampled)
        np.copyto(output, self._padded_actions[: self.action_steps, : self.action_dim])

    def predict_action_chunk(
        self,
        cache: SmolVLAKVCache,
        *,
        noise: np.ndarray | None = None,
        steps: int = 10,
    ) -> np.ndarray:
        """Return an action chunk without modifying the queued action state."""
        output = np.empty_like(self._chunk)
        self._sample_into(cache, output, noise=noise, steps=steps)
        return output

    def refill(
        self,
        cache: SmolVLAKVCache,
        *,
        noise: np.ndarray | None = None,
        steps: int = 10,
    ) -> None:
        """Sample one fresh action chunk from one externally generated VLM cache."""
        self._sample_into(cache, self._chunk, noise=noise, steps=steps)
        self._next_action = 0

    def take_action_into(self, output: np.ndarray) -> np.ndarray:
        """Write the next sampled action into caller-owned storage."""
        self._validate_output(output, (self.action_dim,))
        if self._next_action >= self.action_steps:
            raise RuntimeError("no sampled SmolVLA actions remain; call refill with a fresh VLM cache")
        np.copyto(output, self._chunk[self._next_action])
        self._next_action += 1
        return output

    def take_action(self) -> np.ndarray:
        """Return a copy of the next sampled action for non-hot-path callers."""
        output = np.empty(self.action_dim, dtype=np.float32)
        return self.take_action_into(output)
