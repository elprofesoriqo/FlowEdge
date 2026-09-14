"""Capture a matched full-policy CPU replay benchmark on real PushT observations.

Both paths use the source encoder, processor, observation history, initial noise,
and complete DDIM schedule. Closed-loop success is measured by evaluate.py.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(samples):
    return dict(
        zip(("p50", "p95", "p99"), map(float, np.percentile(samples, (50, 95, 99))))
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-id", default="lerobot/diffusion_pusht")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.steps, args.iterations, args.threads) <= 0 or args.warmup < 0:
        parser.error("invalid step, iteration, thread, or warmup count")
    torch.set_num_threads(args.threads)
    import flowedge
    from .cli import _peak_rss_bytes
    from .diffusion import FlowEdgeDiffusionPolicy
    from .evaluate import PushTRobot
    from .observation import (
        DiffusionObservationEncoder,
        source_stats,
        validate_source_pair,
    )
    from .reference import TorchDiffusionReference

    validate_source_pair(args.checkpoint, args.source)
    startup = perf_counter_ns()
    encoder = DiffusionObservationEncoder(args.source)
    encoder_startup = (perf_counter_ns() - startup) / 1e6
    startup = perf_counter_ns()
    native = FlowEdgeDiffusionPolicy.from_checkpoint(
        args.checkpoint, threads=args.threads
    )
    native_startup = (perf_counter_ns() - startup) / 1e6
    startup = perf_counter_ns()
    reference = TorchDiffusionReference(args.source, native.metadata.condition_dim)
    reference_startup = (perf_counter_ns() - startup) / 1e6
    if (
        reference.config.horizon != native.metadata.horizon
        or reference.config.n_obs_steps != native.metadata.observation_steps
        or reference.config.n_action_steps != native.action_steps
    ):
        raise ValueError("reference and native checkpoint contracts disagree")
    robot = PushTRobot(
        seed=args.seed, max_steps=max(2, native.metadata.observation_steps)
    )
    frames = []
    try:
        robot.reset()
        for _ in range(native.metadata.observation_steps):
            frames.append(robot.observe())
            robot.send_action(np.array([256, 256], dtype=np.float32))
    finally:
        robot.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = args.output.with_suffix(".observations.npz")
    np.savez_compressed(
        fixture,
        **{k: np.stack([frame[k].numpy() for frame in frames]) for k in frames[0]},
    )
    noise = np.random.default_rng(args.seed).standard_normal(
        (native.metadata.horizon, native.action_dim), dtype=np.float32
    )
    samples = {
        name: {"encoder_ms": [], "policy_ms": [], "end_to_end_ms": []}
        for name in ("flowedge", "lerobot")
    }
    max_error = 0.0
    for iteration in range(args.warmup + args.iterations):
        outputs = {}
        order = (("flowedge", native), ("lerobot", reference))
        if iteration % 2:
            order = order[::-1]
        for name, policy in order:
            begin = perf_counter_ns()
            encoder.reset()
            for frame in frames:
                encoder.observe(frame)
            condition = encoder.condition()
            encoded = perf_counter_ns()
            outputs[name] = policy.predict_action_chunk(
                condition, noise, steps=args.steps
            )
            end = perf_counter_ns()
            if iteration >= args.warmup:
                samples[name]["encoder_ms"].append((encoded - begin) / 1e6)
                samples[name]["policy_ms"].append((end - encoded) / 1e6)
                samples[name]["end_to_end_ms"].append((end - begin) / 1e6)
        error = float(np.max(np.abs(outputs["flowedge"] - outputs["lerobot"])))
        max_error = max(error, max_error)
        if not np.allclose(
            outputs["flowedge"], outputs["lerobot"], atol=1e-3, rtol=1e-4
        ):
            raise ValueError(f"full-policy parity failed: max_abs_error={error}")
        print(
            f"replay {iteration + 1}/{args.warmup + args.iterations}: max_abs_error={error:.6g}",
            file=sys.stderr,
        )
    measurements = {}
    rss = _peak_rss_bytes()
    for name, startup_ms in (
        ("flowedge", native_startup),
        ("lerobot", reference_startup),
    ):
        measurements[name] = {
            "status": "measured",
            "startup_ms": startup_ms + encoder_startup,
            **{k: quantiles(v) for k, v in samples[name].items()},
            "throughput_hz": 1000 / float(np.mean(samples[name]["end_to_end_ms"])),
            "rss_mb": None if rss is None else rss / 2**20,
            "rss_reason": "process high-water RSS is unavailable on this host"
            if rss is None
            else "",
            "allocations": {
                "setup": None,
                "hot_path": None,
                "reason": "Python/PyTorch/native process allocations are not instrumented by this runner",
            },
        }
    stats = {
        k: {n: torch.as_tensor(v).tolist() for n, v in values.items()}
        for k, values in source_stats(args.source, encoder.config).items()
    }
    schema = {k: list(v.shape) for k, v in frames[0].items()}
    command = subprocess.list2cmdline(
        [sys.executable, "-m", "flowedge_lerobot.benchmark", *(argv or sys.argv[1:])]
    )
    document = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "sha256": sha256(args.source / "model.safetensors"),
            "converted_sha256": sha256(args.checkpoint),
        },
        "processor": {
            "id": args.model_id,
            "revision": args.revision,
            "stats": stats,
            "config_sha256": sha256(args.source / "config.json"),
        },
        "contract": {
            "observation_schema_hash": hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode()
            ).hexdigest(),
            "observation_steps": native.metadata.observation_steps,
            "action_steps": native.action_steps,
            "action_dim": native.action_dim,
            "batch_size": 1,
            "inference_steps": args.steps,
            "action_units": "dataset",
            "normalization": "minmax",
            "solver": "ddim",
        },
        "hardware": {
            "host": platform.node(),
            "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "threads": args.threads,
            "compiler": flowedge.compiler,
            "build_type": flowedge.build_type,
        },
        "commands": {"flowedge": command, "lerobot": command},
        "measurements": measurements,
        "raw_samples": samples,
        "parity": {"max_abs_error": max_error, "atol": 1e-3, "rtol": 1e-4},
        "fixture": {"file": fixture.name, "sha256": sha256(fixture), "seed": args.seed},
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("torch", "lerobot", "gym-pusht", "numpy")
        },
        "iterations": args.iterations,
        "warmup": args.warmup,
        "limitations": [
            "CPU full-policy replay of one fixed PushT observation history; not closed-loop task success.",
            "End-to-end includes preprocessing, encoder, history and sampling; excludes camera capture, IPC and robot delivery.",
            "RSS is shared process high-water memory with both implementations resident, not per-backend memory.",
            "Allocations are unmeasured; no zero-allocation claim applies to this Python visual pipeline.",
            "Startup excludes interpreter/import time and uses the existing filesystem cache.",
            "Small iteration counts do not establish stable p99 estimates.",
        ],
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
