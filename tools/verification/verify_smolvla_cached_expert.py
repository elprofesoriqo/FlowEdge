#!/usr/bin/env python3
"""Compare FlowEdge's cached-VLM SmolVLA action expert with a real source capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_digest(array) -> str:
    value = hashlib.sha256()
    value.update(str(array.dtype).encode("ascii"))
    value.update(json.dumps(list(array.shape)).encode("ascii"))
    value.update(array.tobytes(order="C"))
    return value.hexdigest()


def required(reference, name: str, dtype, shape: tuple[int, ...]):
    if name not in reference:
        raise ValueError(f"reference is missing {name!r}")
    value = reference[name]
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(
            f"reference {name!r} must have dtype/shape {dtype}/{shape}, got {value.dtype}/{value.shape}"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference", type=Path, help="real source export from export_smolvla_expert_reference.py")
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write portable parity report")
    parser.add_argument(
        "--hidden-tolerance",
        type=float,
        default=1e-1,
        help="maximum BF16 expert-hidden error (default: %(default)g)",
    )
    parser.add_argument(
        "--velocity-tolerance",
        type=float,
        default=8e-2,
        help="maximum cross-runtime BF16 action-velocity error (default: %(default)g)",
    )
    args = parser.parse_args()
    if args.hidden_tolerance <= 0.0 or args.velocity_tolerance <= 0.0:
        parser.error("tolerances must be positive")
    sys.path.insert(0, str(args.module_path))

    try:
        import flowedge
        import numpy as np

        reference = np.load(args.reference, allow_pickle=False)
        metadata_engine = flowedge.Engine(str(args.checkpoint), threads=0)
        metadata = metadata_engine.model_metadata
        if metadata["architecture"] != 7 or metadata["n_layers"] != 16 or metadata["action_dim"] != 32:
            raise RuntimeError("checkpoint is not the supported 16-layer SmolVLA action expert")
        noisy_actions = required(reference, "noisy_actions", np.dtype("float32"), (50, 32))
        timestep = required(reference, "timestep", np.dtype("float32"), (1,))
        if "prefix_mask" not in reference:
            raise ValueError("reference is missing 'prefix_mask'")
        prefix_mask = reference["prefix_mask"]
        if prefix_mask.dtype != np.dtype("uint8") or prefix_mask.ndim != 1:
            raise ValueError("reference prefix_mask must be a uint8 vector")
        prefix_length = prefix_mask.size
        if prefix_length == 0 or prefix_length > 512 or not np.isin(prefix_mask, (0, 1)).all():
            raise ValueError("reference prefix_mask must contain 1..512 one/zero entries")
        if not prefix_mask.any():
            raise ValueError("reference prefix_mask must contain at least one valid token")
        prefix_keys = required(reference, "prefix_keys", np.dtype("float32"), (16, prefix_length, 320))
        prefix_values = required(reference, "prefix_values", np.dtype("float32"), (16, prefix_length, 320))
        expected_hidden = required(reference, "expected_hidden", np.dtype("float32"), (50, 720))
        expected_velocity = required(reference, "expected_velocity", np.dtype("float32"), (50, 32))
        suffix = metadata_engine.smolvla_embed_suffix(noisy_actions, float(timestep[0]))
        actual_hidden = metadata_engine.smolvla_run_expert(suffix, prefix_keys, prefix_values, prefix_mask)
        split_velocity = metadata_engine.smolvla_project_actions(actual_hidden)
        actual_velocity = metadata_engine.smolvla_denoise(
            noisy_actions, float(timestep[0]), prefix_keys, prefix_values, prefix_mask
        )
        if not np.array_equal(actual_velocity, split_velocity):
            raise RuntimeError("fused SmolVLA denoise output differs from the split native boundary")
        if not np.isfinite(actual_hidden).all() or not np.isfinite(actual_velocity).all():
            raise RuntimeError("FlowEdge action expert emitted non-finite values")
        hidden_error = np.abs(actual_hidden - expected_hidden)
        velocity_error = np.abs(actual_velocity - expected_velocity)
        hidden_max = float(hidden_error.max())
        velocity_max = float(velocity_error.max())
        if hidden_max > args.hidden_tolerance or velocity_max > args.velocity_tolerance:
            raise RuntimeError(
                f"cached expert parity exceeded tolerance: hidden={hidden_max}, velocity={velocity_max}"
            )
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"SmolVLA cached-expert verification failed: {error}", file=sys.stderr)
        return 2

    result = {
        "schema_version": 1,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": digest(args.checkpoint),
        "reference": args.reference.name,
        "reference_sha256": digest(args.reference),
        "source_noisy_actions_sha256": array_digest(noisy_actions),
        "source_prefix_keys_sha256": array_digest(prefix_keys),
        "source_prefix_values_sha256": array_digest(prefix_values),
        "prefix_length": int(prefix_length),
        "expert_layers": 16,
        "expert_width": 720,
        "key_value_width": 320,
        "hidden_max_absolute_error": hidden_max,
        "hidden_mean_absolute_error": float(hidden_error.mean()),
        "hidden_tolerance": args.hidden_tolerance,
        "velocity_max_absolute_error": velocity_max,
        "velocity_mean_absolute_error": float(velocity_error.mean()),
        "velocity_tolerance": args.velocity_tolerance,
        "status": "passed",
        "scope": (
            "real captured VLM K/V cache replay through the native SmolVLA action expert; "
            "not VLM/preprocessor parity, latency, or policy-quality evaluation"
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
