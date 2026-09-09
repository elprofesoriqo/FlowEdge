"""LeRobot Diffusion Policy action contract over the FlowEdge Python API.

Observation encoding and LeRobot feature preprocessing stay outside this package. The adapter
owns only the deployment boundary: validating a flattened condition, invoking FlowEdge, and
slicing the action horizon according to LeRobot's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class DiffusionEngine(Protocol):
    """Minimum engine surface required by the adapter."""

    @property
    def diffusion_metadata(self) -> dict[str, Any]: ...

    def sample_diffusion(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        steps: int,
        scheduler: str,
        seed: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class DiffusionActionContract:
    """Fixed-shape action boundary shared with a converted FlowEdge checkpoint."""

    condition_dim: int
    action_dim: int
    horizon: int
    action_steps: int
    observation_steps: int

    @property
    def action_start(self) -> int:
        """Return the first horizon element consumed by LeRobot."""
        return self.observation_steps - 1

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "DiffusionActionContract":
        required = (
            "condition_dim",
            "action_dim",
            "horizon",
            "action_steps",
            "observation_steps",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"missing diffusion metadata: {', '.join(missing)}")
        contract = cls(*(int(metadata[key]) for key in required))
        if contract.action_start < 0 or contract.action_start + contract.action_steps > contract.horizon:
            raise ValueError("LeRobot action slice is outside the diffusion horizon")
        if min(contract.condition_dim, contract.action_dim, contract.horizon, contract.action_steps) <= 0:
            raise ValueError("diffusion dimensions must be positive")
        if contract.observation_steps <= 0:
            raise ValueError("observation_steps must be positive")
        return contract


class FlowEdgeDiffusionPolicy:
    """Deployment adapter for a converted LeRobot Diffusion Policy checkpoint.

    The returned action chunk is already in dataset action units. LeRobot's image/state encoder,
    feature normalization, and robot safety limits remain the caller's responsibility.
    """

    def __init__(self, engine: DiffusionEngine):
        self._engine = engine
        self.contract = DiffusionActionContract.from_metadata(engine.diffusion_metadata)

    @classmethod
    def from_checkpoint(cls, path: str, *, threads: int | None = None) -> "FlowEdgeDiffusionPolicy":
        """Load a converted checkpoint without making FlowEdge a LeRobot dependency."""
        import flowedge

        engine = flowedge.Engine(path, threads=threads)
        return cls(engine)

    @property
    def metadata(self) -> DiffusionActionContract:
        return self.contract

    @property
    def action_dim(self) -> int:
        return self.contract.action_dim

    @property
    def action_steps(self) -> int:
        return self.contract.action_steps

    def reset(self) -> None:
        """Reset the deployment adapter; the current diffusion engine is stateless per call."""

    def predict_action_chunk(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        *,
        steps: int = 10,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> np.ndarray:
        """Return LeRobot's executable action chunk from a flattened condition."""
        normalized_condition = np.ascontiguousarray(condition, dtype=np.float32).reshape(-1)
        normalized_noise = np.ascontiguousarray(noise, dtype=np.float32)
        expected_noise = (self.contract.horizon, self.contract.action_dim)
        if normalized_condition.size != self.contract.condition_dim:
            raise ValueError(
                f"condition must contain {self.contract.condition_dim} float32 values"
            )
        if normalized_noise.shape != expected_noise:
            raise ValueError(f"noise must have shape {expected_noise}")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if scheduler not in {"ddim", "ddpm"}:
            raise ValueError("scheduler must be one of: ddim, ddpm")
        horizon = np.asarray(
            self._engine.sample_diffusion(
                normalized_condition, normalized_noise, steps, scheduler, int(seed)
            ),
            dtype=np.float32,
        )
        if horizon.shape != expected_noise:
            raise ValueError(f"FlowEdge returned horizon with shape {horizon.shape}, expected {expected_noise}")
        start = self.contract.action_start
        stop = start + self.contract.action_steps
        return np.ascontiguousarray(horizon[start:stop])

    def select_action(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        *,
        steps: int = 10,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> np.ndarray:
        """Return the first action in the executable chunk for single-step loops."""
        return self.predict_action_chunk(
            condition, noise, steps=steps, scheduler=scheduler, seed=seed
        )[0]
