#!/usr/bin/env python3
"""Validate a pinned upstream SmolVLA checkpoint before conversion.

This deliberately reads only the safetensors header.  It is a provenance and
layout gate, not a converter and not an inference benchmark: FlowEdge does not
yet execute the upstream VLM encoder or an end-to-end SmolVLA policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


PINNED_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
EXPECTED = {
    "action_dim": 6,
    "chunk_size": 50,
    "n_action_steps": 50,
    "num_steps": 10,
    "num_vlm_layers": 16,
    "self_attn_every_n_layers": 2,
    "expert_width_multiplier": 0.75,
}


def header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        encoded_size = handle.read(8)
        if len(encoded_size) != 8:
            raise ValueError("safetensors file has no complete header size")
        size = struct.unpack("<Q", encoded_size)[0]
        if size == 0 or size > 64 * 1024 * 1024:
            raise ValueError("safetensors header size is invalid")
        payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("safetensors header is incomplete")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("safetensors header is not an object")
    return value


def shape(entry: dict[str, Any]) -> list[int]:
    value = entry.get("shape")
    if not isinstance(value, list) or not all(isinstance(x, int) and x > 0 for x in value):
        raise ValueError("tensor shape is malformed")
    return value


def expect_tensor(tensors: dict[str, Any], name: str, expected_shape: list[int], dtype: str) -> str | None:
    entry = tensors.get(name)
    if not isinstance(entry, dict):
        return f"missing tensor: {name}"
    try:
        actual_shape = shape(entry)
    except ValueError:
        return f"malformed tensor: {name}"
    if actual_shape != expected_shape:
        return f"shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"
    if entry.get("dtype") != dtype:
        return f"dtype mismatch for {name}: expected {dtype}, got {entry.get('dtype')}"
    return None


def validate(config: dict[str, Any], tensors: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("type") != "smolvla":
        errors.append("config.type must be smolvla")
    for key, expected in EXPECTED.items():
        if key == "action_dim":
            continue
        actual = config.get(key)
        if actual != expected:
            errors.append(f"config.{key} must be {expected!r}, got {actual!r}")
    action = config.get("output_features", {}).get("action", {}).get("shape")
    if action != [EXPECTED["action_dim"]]:
        errors.append("config output action shape must be [6]")

    root = {
        "model.action_in_proj.weight": ([720, 32], "F32"),
        "model.action_in_proj.bias": ([720], "F32"),
        "model.action_out_proj.weight": ([32, 720], "F32"),
        "model.action_out_proj.bias": ([32], "F32"),
        "model.action_time_mlp_in.weight": ([720, 1440], "F32"),
        "model.action_time_mlp_in.bias": ([720], "F32"),
        "model.action_time_mlp_out.weight": ([720, 720], "F32"),
        "model.action_time_mlp_out.bias": ([720], "F32"),
        "model.state_proj.weight": ([960, 32], "F32"),
        "model.state_proj.bias": ([960], "F32"),
    }
    for name, (expected_shape, dtype) in root.items():
        error = expect_tensor(tensors, name, expected_shape, dtype)
        if error:
            errors.append(error)

    for layer in range(EXPECTED["num_vlm_layers"]):
        prefix = f"model.vlm_with_expert.lm_expert.layers.{layer}."
        layer_tensors = {
            "input_layernorm.weight": ([720], "BF16"),
            "post_attention_layernorm.weight": ([720], "BF16"),
            "self_attn.q_proj.weight": ([960, 720], "BF16"),
            "self_attn.o_proj.weight": ([720, 960], "BF16"),
            "mlp.gate_proj.weight": ([2048, 720], "BF16"),
            "mlp.up_proj.weight": ([2048, 720], "BF16"),
            "mlp.down_proj.weight": ([720, 2048], "BF16"),
        }
        # Even layers attend to action tokens; odd layers cross-attend to
        # VLM K/V cache, which changes only the projection input width.
        kv_width = 720 if layer % EXPECTED["self_attn_every_n_layers"] == 0 else 320
        layer_tensors["self_attn.k_proj.weight"] = ([320, kv_width], "BF16" if kv_width == 720 else "F32")
        layer_tensors["self_attn.v_proj.weight"] = ([320, kv_width], "BF16" if kv_width == 720 else "F32")
        for suffix, (expected_shape, dtype) in layer_tensors.items():
            error = expect_tensor(tensors, prefix + suffix, expected_shape, dtype)
            if error:
                errors.append(error)
    return errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--output", type=Path, help="write the verified JSON manifest")
    parser.add_argument("--hash", action="store_true", help="include SHA-256 after a complete download")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        tensors = header(args.checkpoint)
        tensors.pop("__metadata__", None)
        errors = validate(config, tensors)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": "lerobot/smolvla_base",
        "revision": args.revision,
        # Keep artifacts portable and avoid embedding a contributor's absolute
        # workspace path. The model/revision/digest above identify the input.
        "checkpoint": args.checkpoint.name,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "config": {
            "action_dim": EXPECTED["action_dim"],
            "chunk_size": EXPECTED["chunk_size"],
            "flow_steps": EXPECTED["num_steps"],
            "expert_layers": EXPECTED["num_vlm_layers"],
            "expert_width": 720,
            "attention": "GQA self/cross attention",
            "norm": "RMSNorm",
            "mlp": "SwiGLU",
        },
        "tensor_count": len(tensors),
        "supported_by_flowedge": False,
        "status": "preflight-valid" if not errors else "preflight-invalid",
        "errors": errors,
    }
    if args.hash and not errors:
        report["sha256"] = sha256(args.checkpoint)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
