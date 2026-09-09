"""Small, backend-neutral rollout loop for LeRobot robot adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .diffusion import FlowEdgeDiffusionPolicy


class RolloutRobot(Protocol):
    """Minimal robot surface needed by the deployment loop."""

    def reset(self) -> None: ...

    def observe(self) -> np.ndarray: ...

    def send_action(self, action: np.ndarray) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class RolloutResult:
    """Bounded result returned after a rollout completes or fails."""

    steps: int
    stopped: bool


def run_rollout(
    policy: FlowEdgeDiffusionPolicy,
    robot: RolloutRobot,
    encode_condition: Callable[[np.ndarray], np.ndarray],
    *,
    steps: int,
    seed: int = 0,
    diffusion_steps: int = 10,
    scheduler: str = "ddim",
) -> RolloutResult:
    """Run a bounded single-action loop suitable for SO-100/SO-101 integration shims.

    ``encode_condition`` is injected so LeRobot owns camera/state feature encoding and
    normalization. ``stop`` is always called, including when an operation fails.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be positive")

    policy.reset()
    rng = np.random.default_rng(seed)
    completed = 0
    try:
        robot.reset()
        for _ in range(steps):
            condition = encode_condition(robot.observe())
            noise_shape = (policy.metadata.horizon, policy.action_dim)
            noise = np.asarray(rng.standard_normal(noise_shape), dtype=np.float32)
            action = policy.select_action(
                condition,
                noise,
                steps=diffusion_steps,
                scheduler=scheduler,
                seed=seed + completed,
            )
            robot.send_action(action)
            completed += 1
    finally:
        robot.stop()
    return RolloutResult(steps=completed, stopped=True)
