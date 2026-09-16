#!/usr/bin/env python3
"""Verify the real SmolVLA action/time suffix projection against PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timestep", type=float, default=0.37)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.chunk_size > 512:
        parser.error("chunk-size must be between 1 and 512")
    sys.path.insert(0, str(args.module_path))

    try:
        import flowedge
        import numpy as np
        import torch
        from safetensors.torch import load_file

        weights = load_file(str(args.checkpoint))
        action_in = weights["model.action_in_proj.weight"]
        action_dim = int(action_in.shape[1])
        expert_width = int(action_in.shape[0])
        rng = np.random.default_rng(args.seed)
        actions = np.ascontiguousarray(
            rng.standard_normal((args.chunk_size, action_dim), dtype=np.float32)
        )
        engine = flowedge.Engine(str(args.checkpoint), threads=0)
        actual = engine.smolvla_embed_suffix(actions, args.timestep)

        action_emb = torch.nn.functional.linear(
            torch.from_numpy(actions),
            weights["model.action_in_proj.weight"],
            weights["model.action_in_proj.bias"],
        )
        half = expert_width // 2
        fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float64)
        period = 0.004 * (4.0 / 0.004) ** fraction
        scale = 2.0 * math.pi / period
        time = torch.cat(
            [torch.sin(scale * args.timestep), torch.cos(scale * args.timestep)]
        ).to(dtype=action_emb.dtype)
        fused = torch.cat([action_emb, time.expand(args.chunk_size, -1)], dim=1)
        expected = torch.nn.functional.silu(
            torch.nn.functional.linear(
                fused,
                weights["model.action_time_mlp_in.weight"],
                weights["model.action_time_mlp_in.bias"],
            )
        )
        expected = torch.nn.functional.linear(
            expected,
            weights["model.action_time_mlp_out.weight"],
            weights["model.action_time_mlp_out.bias"],
        ).numpy()
        max_error = float(np.max(np.abs(actual - expected)))
        mean_error = float(np.mean(np.abs(actual - expected)))
        if max_error > 5e-5:
            raise RuntimeError(f"suffix projection parity exceeded tolerance: {max_error}")
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"SmolVLA action-expert verification failed: {error}", file=sys.stderr)
        return 2

    result = {
        "schema_version": 1,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "chunk_size": args.chunk_size,
        "action_dim": action_dim,
        "expert_width": expert_width,
        "timestep": args.timestep,
        "seed": args.seed,
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "tolerance": 5e-5,
        "status": "passed",
        "scope": (
            "real SmolVLA action_in_proj, sinusoidal timestep embedding, and action-time MLP "
            "against PyTorch; VLM preprocessing and interleaved expert attention are not included"
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
