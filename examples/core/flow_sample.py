#!/usr/bin/env python3
"""Run deterministic FlowEdge flow-policy inference from Python."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _load_flowedge():
    if os.name == "nt":
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            runtime = Path(entry) / "libwinpthread-1.dll"
            if runtime.is_file():
                os.add_dll_directory(str(runtime.parent))
                break
    try:
        import flowedge
    except ImportError as error:
        raise RuntimeError(
            "install FlowEdge first with: python -m pip install ."
        ) from error
    return flowedge


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="FlowEdge safetensors checkpoint")
    parser.add_argument(
        "solver", choices=("euler", "heun", "rk4"), nargs="?", default="euler"
    )
    parser.add_argument("steps", type=int, nargs="?", default=10)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("steps must be positive")

    try:
        engine = _load_flowedge().Engine(args.model)
        action_dim = engine.action_dim
        if action_dim == 0:
            raise RuntimeError("checkpoint has no flow head")
        prefix = np.array([1, 2, 3, 4], dtype=np.int32)
        noise = np.sin(np.arange(action_dim, dtype=np.float32) * 0.3)
        action = engine.sample(prefix, noise, args.steps, args.solver)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    nfe = args.steps * {"euler": 1, "heun": 2, "rk4": 4}[args.solver]
    preview = ", ".join(f"{action[index % action_dim]:.4f}" for index in range(3))
    print(
        f"action_dim={action_dim} solver={args.solver} NFE={nfe} action[0..2]={preview}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
