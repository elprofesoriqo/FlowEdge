#!/usr/bin/env python3
"""Compare FlowEdge's cached-VLM SmolVLA Euler solve with a real source capture."""

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
    parser.add_argument("reference", type=Path, help="real source export with --trajectory-output")
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write portable parity report")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3e-2,
        help="maximum cross-runtime BF16 Euler action error (default: %(default)g)",
    )
    args = parser.parse_args()
    if args.tolerance <= 0.0:
        parser.error("tolerance must be positive")
    sys.path.insert(0, str(args.module_path))

    try:
        import flowedge
        import numpy as np

        reference = np.load(args.reference, allow_pickle=False)
        engine = flowedge.Engine(str(args.checkpoint), threads=0)
        metadata = engine.model_metadata
        if metadata["architecture"] != 7 or metadata["n_layers"] != 16 or metadata["action_dim"] != 32:
            raise RuntimeError("checkpoint is not the supported 16-layer SmolVLA action expert")
        initial_noise = required(reference, "initial_noise", np.dtype("float32"), (50, 32))
        expected_actions = required(reference, "expected_actions", np.dtype("float32"), (50, 32))
        steps_array = required(reference, "steps", np.dtype("int64"), (1,))
        steps = int(steps_array[0])
        if steps == 0 or steps > 100:
            raise ValueError("reference Euler step count must be in [1, 100]")
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
        if not (
            np.isfinite(initial_noise).all()
            and np.isfinite(prefix_keys).all()
            and np.isfinite(prefix_values).all()
            and np.isfinite(expected_actions).all()
        ):
            raise ValueError("reference contains non-finite values")
        actual_actions = engine.smolvla_sample(
            initial_noise, prefix_keys, prefix_values, prefix_mask, steps=steps
        )
        if not np.isfinite(actual_actions).all():
            raise RuntimeError("FlowEdge Euler solve emitted non-finite values")
        error = np.abs(actual_actions - expected_actions)
        max_error = float(error.max())
        if max_error > args.tolerance:
            raise RuntimeError(
                f"cached Euler trajectory parity exceeded tolerance: actions={max_error}"
            )
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"SmolVLA cached trajectory verification failed: {error}", file=sys.stderr)
        return 2

    result = {
        "schema_version": 1,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": digest(args.checkpoint),
        "reference": args.reference.name,
        "reference_sha256": digest(args.reference),
        "source_initial_noise_sha256": array_digest(initial_noise),
        "source_prefix_keys_sha256": array_digest(prefix_keys),
        "source_prefix_values_sha256": array_digest(prefix_values),
        "prefix_length": int(prefix_length),
        "expert_layers": 16,
        "expert_width": 720,
        "key_value_width": 320,
        "steps": steps,
        "actions_max_absolute_error": max_error,
        "actions_mean_absolute_error": float(error.mean()),
        "tolerance": args.tolerance,
        "status": "passed",
        "scope": (
            "real captured VLM K/V cache replay through the native SmolVLA action-expert Euler "
            "solve; not VLM/preprocessor parity, latency, or policy-quality evaluation"
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
