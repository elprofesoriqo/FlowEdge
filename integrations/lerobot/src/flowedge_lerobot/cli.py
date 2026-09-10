"""Simulator-first command line entry point for LeRobot rollout smoke tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

import numpy as np

from .diffusion import FlowEdgeDiffusionPolicy
from .rollout import RolloutResult, run_rollout


class _SimulatedRobot:
    """Deterministic robot seam that keeps observation and action buffers owned."""

    def __init__(self, condition_dim: int) -> None:
        self._observation = np.zeros(condition_dim, dtype=np.float32)
        self.actions = 0
        self.stopped = False

    def reset(self) -> None:
        self.actions = 0
        self.stopped = False

    def observe(self) -> np.ndarray:
        return self._observation

    def send_action(self, action: np.ndarray) -> None:
        del action
        self.actions += 1

    def stop(self) -> None:
        self.stopped = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowedge-lerobot-rollout",
        description="Run a bounded simulator smoke test for a converted LeRobot policy.",
    )
    parser.add_argument(
        "checkpoint", help="FlowEdge-converted Diffusion Policy checkpoint"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="control-loop steps (default: 10)"
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=10, help="denoising steps (default: 10)"
    )
    parser.add_argument("--scheduler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--period-ms",
        type=float,
        default=None,
        help="deadline period for missed-step counts",
    )
    return parser


def run_simulator(
    policy: FlowEdgeDiffusionPolicy,
    *,
    steps: int,
    diffusion_steps: int,
    scheduler: str,
    seed: int,
    period_ms: float | None = None,
) -> RolloutResult:
    """Run the built-in simulator seam; hardware adapters stay outside this package."""

    robot = _SimulatedRobot(policy.metadata.condition_dim)
    return run_rollout(
        policy,
        robot,
        lambda observation: observation,
        steps=steps,
        seed=seed,
        diffusion_steps=diffusion_steps,
        scheduler=scheduler,
        period_ms=period_ms,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = FlowEdgeDiffusionPolicy.from_checkpoint(
        args.checkpoint, threads=args.threads
    )
    result = run_simulator(
        policy,
        steps=args.steps,
        diffusion_steps=args.diffusion_steps,
        scheduler=args.scheduler,
        seed=args.seed,
        period_ms=args.period_ms,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
