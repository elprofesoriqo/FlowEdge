#!/usr/bin/env python3
"""Export an upstream SmolVLA action chunk for a captured real observation.

The capture is deliberately required. This tool never creates images, language
tokens, state, or initial flow noise: parity evidence must begin with a real
LeRobot observation processed by the source pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


DEFAULT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


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


def required(capture, name: str, dtype, shape: tuple[int, ...]):
    if name not in capture:
        raise ValueError(f"capture is missing {name!r}")
    value = capture[name]
    if value.dtype != dtype:
        raise ValueError(f"capture {name!r} must be {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"capture {name!r} must have shape {shape}, got {tuple(value.shape)}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local lerobot/smolvla_base directory")
    parser.add_argument("capture", type=Path, help="real source-pipeline .npz capture")
    parser.add_argument("--output", type=Path, required=True, help="write source action chunk .npz")
    parser.add_argument("--manifest", type=Path, help="write JSON provenance beside the action chunk")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        import lerobot
        import transformers
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if not args.capture.is_file():
            raise ValueError(f"capture does not exist: {args.capture}")
        policy = SmolVLAPolicy.from_pretrained(args.model).eval()
        batch_size = 1
        state_dim = policy.config.state_feature.shape[0]
        action_dim = policy.config.action_feature.shape[0]
        capture = np.load(args.capture, allow_pickle=False)
        state = required(capture, "observation.state", np.dtype("float32"), (batch_size, state_dim))
        tokens = required(
            capture,
            "observation.language.tokens",
            np.dtype("int64"),
            (batch_size, policy.config.tokenizer_max_length),
        )
        attention_mask = required(
            capture,
            "observation.language.attention_mask",
            np.dtype("int64"),
            (batch_size, policy.config.tokenizer_max_length),
        )
        noise = required(
            capture,
            "noise",
            np.dtype("float32"),
            (batch_size, policy.config.chunk_size, policy.config.max_action_dim),
        )
        if not np.isfinite(state).all() or not np.isfinite(noise).all():
            raise ValueError("capture state and noise must be finite")
        if not np.isin(attention_mask, (0, 1)).all():
            raise ValueError("capture language attention mask must contain only 0 or 1")
        batch = {
            "observation.state": torch.from_numpy(state),
            "observation.language.tokens": torch.from_numpy(tokens),
            "observation.language.attention_mask": torch.from_numpy(attention_mask),
        }
        for key, feature in policy.config.image_features.items():
            channels, height, width = feature.shape
            image = required(capture, key, np.dtype("float32"), (batch_size, channels, height, width))
            if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
                raise ValueError(f"capture {key!r} must be finite RGB in [0, 1]")
            batch[key] = torch.from_numpy(image)
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise))
        action = actions.detach().cpu().numpy()
        expected_shape = (batch_size, policy.config.n_action_steps, action_dim)
        if action.dtype != np.float32 or tuple(action.shape) != expected_shape:
            raise RuntimeError(f"source action has shape/dtype {action.shape}/{action.dtype}, expected {expected_shape}/float32")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, action_chunk=action)
        report = {
            "schema_version": 1,
            "model": "lerobot/smolvla_base",
            "revision": args.revision,
            "source_capture_sha256": digest(args.capture),
            "source_noise_sha256": array_digest(noise),
            "source_action_sha256": array_digest(action),
            "action_shape": list(action.shape),
            "action_dtype": str(action.dtype),
            "lerobot_version": lerobot.__version__,
            "transformers_version": transformers.__version__,
            "status": "passed",
            "scope": (
                "upstream SmolVLA source action chunk for a supplied real capture; "
                "not FlowEdge inference, latency, or policy-quality evidence"
            ),
        }
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"SmolVLA reference export failed: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
