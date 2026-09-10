"""Small, backend-neutral rollout loop for LeRobot robot adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns
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
    elapsed_ms: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    missed_deadlines: int = 0


MAX_ROLLOUT_STEPS = 100_000


def run_rollout(
    policy: FlowEdgeDiffusionPolicy,
    robot: RolloutRobot,
    encode_condition: Callable[[np.ndarray], np.ndarray],
    *,
    steps: int,
    seed: int = 0,
    diffusion_steps: int = 10,
    scheduler: str = "ddim",
    period_ms: float | None = None,
) -> RolloutResult:
    """Run a bounded single-action loop suitable for SO-100/SO-101 integration shims.

    ``encode_condition`` is injected so LeRobot owns camera/state feature encoding and
    normalization. ``stop`` is always called, including when an operation fails.
    """

    if steps <= 0 or steps > MAX_ROLLOUT_STEPS:
        raise ValueError(f"steps must be from 1 through {MAX_ROLLOUT_STEPS}")
    if diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be positive")
    if period_ms is not None and (not isfinite(period_ms) or period_ms <= 0):
        raise ValueError("period_ms must be positive")

    policy.reset()
    rng = np.random.default_rng(seed)
    noise = np.empty((policy.metadata.horizon, policy.action_dim), dtype=np.float32)
    action = np.empty(policy.action_dim, dtype=np.float32)
    durations_ns = np.empty(steps, dtype=np.int64)
    rollout_start = perf_counter_ns()
    completed = 0
    try:
        robot.reset()
        for _ in range(steps):
            step_start = perf_counter_ns()
            condition = encode_condition(robot.observe())
            rng.standard_normal(noise.shape, dtype=np.float32, out=noise)
            policy.select_action_into(
                condition,
                noise,
                action,
                steps=diffusion_steps,
                scheduler=scheduler,
                seed=seed + completed,
            )
            robot.send_action(action)
            durations_ns[completed] = perf_counter_ns() - step_start
            completed += 1
    finally:
        robot.stop()
    elapsed_ns = perf_counter_ns() - rollout_start
    if completed == 0:
        return RolloutResult(steps=0, stopped=True, elapsed_ms=elapsed_ns / 1e6)
    samples = durations_ns[:completed]
    p50, p99 = np.percentile(samples, (50, 99))
    deadline_ns = None if period_ms is None else period_ms * 1e6
    return RolloutResult(
        steps=completed,
        stopped=True,
        elapsed_ms=elapsed_ns / 1e6,
        p50_ms=float(p50 / 1e6),
        p99_ms=float(p99 / 1e6),
        max_ms=float(samples.max() / 1e6),
        missed_deadlines=0
        if deadline_ns is None
        else int(np.count_nonzero(samples > deadline_ns)),
    )
