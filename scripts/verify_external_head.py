#!/usr/bin/env python3

"""Smoke-test a flow-head-only checkpoint and cooperative solver execution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from safetensors.numpy import save_file


def load_extension(extension_dir: Path):
    sys.path.insert(0, str(extension_dir.resolve()))
    if os.name == "nt":
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            winpthread = Path(entry) / "libwinpthread-1.dll"
            if winpthread.is_file():
                os.add_dll_directory(str(winpthread.parent))
                break
    import flowedge

    return flowedge


def weights(seed: int = 7) -> dict[str, np.ndarray]:
    hidden, action, time, condition = 48, 8, 16, 32
    rng = np.random.default_rng(seed)

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (rng.standard_normal((rows, columns)) * 0.01).astype(np.float32)

    return {
        "flow.in_proj.weight": matrix(hidden, action),
        "flow.time_proj.weight": matrix(hidden, time),
        "flow.cond_proj.weight": matrix(hidden, condition),
        "flow.out_proj.weight": matrix(action, hidden),
    }


def mamba_weights(seed: int = 11) -> dict[str, np.ndarray]:
    vocab, model, inner, state, conv, rank = 16, 8, 8, 2, 3, 1
    rng = np.random.default_rng(seed)

    def matrix(rows: int, columns: int, scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal((rows, columns)) * scale).astype(np.float32)

    return {
        "backbone.embeddings.weight": matrix(vocab, model),
        "backbone.norm_f.weight": np.ones(model, dtype=np.float32),
        "backbone.layers.0.norm.weight": np.ones(model, dtype=np.float32),
        "backbone.layers.0.mixer.in_proj.weight": matrix(2 * inner, model),
        "backbone.layers.0.mixer.conv1d.weight": matrix(inner, conv).reshape(inner, 1, conv),
        "backbone.layers.0.mixer.conv1d.bias": np.zeros(inner, dtype=np.float32),
        "backbone.layers.0.mixer.x_proj.weight": matrix(rank + (2 * state), inner),
        "backbone.layers.0.mixer.dt_proj.weight": matrix(inner, rank),
        "backbone.layers.0.mixer.dt_proj.bias": np.full(inner, 0.1, dtype=np.float32),
        "backbone.layers.0.mixer.A_log": matrix(inner, state),
        "backbone.layers.0.mixer.D": np.full(inner, 0.2, dtype=np.float32),
        "backbone.layers.0.mixer.out_proj.weight": matrix(model, inner),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("extension_dir", nargs="?", default="build")
    args = parser.parse_args()
    flowedge = load_extension(Path(args.extension_dir))

    with tempfile.TemporaryDirectory(prefix="flowedge-") as temp_dir:
        checkpoint = Path(temp_dir) / "external-flow-head.safetensors"
        save_file(weights(), checkpoint)
        engine = flowedge.Engine(str(checkpoint))

        condition = np.linspace(-0.1, 0.1, engine.condition_dim, dtype=np.float32)
        noise = np.linspace(-0.2, 0.2, engine.action_dim, dtype=np.float32)
        direct = np.empty(engine.action_dim, dtype=np.float32)
        engine.sample_condition(condition, noise, direct, 6, "heun")

        engine.flow_begin(condition, noise, 6, "heun")
        resumed = np.empty(engine.action_dim, dtype=np.float32)
        remaining = engine.flow_advance(resumed, 2)
        if remaining != 4:
            raise RuntimeError(f"expected four remaining steps, got {remaining}")
        remaining = engine.flow_advance(resumed, 99)
        if remaining != 0 or not np.array_equal(direct, resumed):
            raise RuntimeError("resumable and monolithic solvers diverged")

        mamba_checkpoint = Path(temp_dir) / "streaming-mamba.safetensors"
        save_file(mamba_weights(), mamba_checkpoint)
        mamba = flowedge.Engine(str(mamba_checkpoint))
        tokens = np.array([1, 4, 7, 2], dtype=np.int32)
        batch = mamba.run(tokens)
        streamed = np.stack([mamba.step(int(token)) for token in tokens])
        if not np.allclose(batch, streamed, rtol=3e-6, atol=3e-6):
            raise RuntimeError("streaming Mamba output differs from batch output")

        batch_into = np.empty_like(batch)
        mamba.run_into(tokens, batch_into)
        mamba.reset()
        streamed_into = np.empty_like(streamed)
        for index, token in enumerate(tokens):
            mamba.step_into(int(token), streamed_into[index])
        if not np.array_equal(batch, batch_into) or not np.array_equal(streamed, streamed_into):
            raise RuntimeError("caller-owned output path differs from convenience calls")

        mamba.reset()
        mamba.step(3)
        snapshot = mamba.decode_state()
        branch_a = mamba.step(5)
        mamba.restore_decode_state(snapshot)
        branch_b = mamba.step(5)
        if not np.array_equal(branch_a, branch_b):
            raise RuntimeError("restored streaming state did not reproduce the branch")

        print(
            f"PASS condition_dim={engine.condition_dim} action_dim={engine.action_dim} "
            "resumable_exact=true streaming_exact=true"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
