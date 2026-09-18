#!/usr/bin/env python3
"""Compare real GPT-2 full-prefix and streaming decode with FlowEdge.

This is a conversion/runtime smoke check, not a policy-quality or latency
benchmark. It intentionally uses a downloaded checkpoint instead of generated
weights and checks the native FlowEdge executable against the source model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import GPT2Model


DEFAULT_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_weight_digest(source: Path) -> str:
    for name in ("pytorch_model.bin", "model.safetensors", "model.bin"):
        candidate = source / name
        if candidate.is_file():
            return sha256(candidate)
    raise OSError(f"no pytorch_model.bin or model.safetensors in {source}")


def runtime_path(path: Path, executable: Path) -> str:
    """Adapt a WSL path only when invoking a Windows FlowEdge executable."""
    if executable.suffix.lower() != ".exe" or not str(path).startswith("/mnt/"):
        return str(path)
    completed = subprocess.run(
        ["wslpath", "-w", str(path)], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise OSError(f"cannot convert {path} to a Windows path")
    return completed.stdout.strip()


def run_flowedge(binary: Path, converted: Path, tokens: list[int], stream: bool) -> list[float]:
    command = [str(binary), runtime_path(converted, binary), "--json"]
    if stream:
        command.append("--stream")
    command.extend(str(token) for token in tokens)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "FlowEdge execution failed: " + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        result = json.loads(completed.stdout)
        outputs = result["outputs"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("FlowEdge did not emit a valid JSON output") from error
    if result.get("mode") != ("stream" if stream else "prefill"):
        raise RuntimeError("FlowEdge reported an unexpected execution mode")
    if not isinstance(result.get("d_model"), int) or result["d_model"] <= 0:
        raise RuntimeError("FlowEdge reported an invalid hidden width")
    if not isinstance(outputs, list) or len(outputs) != len(tokens) * result["d_model"]:
        raise RuntimeError("FlowEdge reported an invalid output length")
    return [float(value) for value in outputs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local GPT-2-style source directory")
    parser.add_argument("converted", type=Path, help="FlowEdge converted safetensors file")
    parser.add_argument(
        "--binary", type=Path, required=True, help="built transformer_forward executable"
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face id recorded in the JSON (tiny-gpt2 is the CI fixture; "
        "openai-community/gpt2 is optional local evidence)",
    )
    parser.add_argument("--tolerance", type=float, default=5e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    try:
        model = GPT2Model.from_pretrained(args.source).eval()
        prefix = [1, 2, 3, 4]
        tokens = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            reference = model(input_ids=tokens).last_hidden_state.flatten().tolist()
        prefill = run_flowedge(args.binary, args.converted, prefix, stream=False)
        streaming = run_flowedge(args.binary, args.converted, prefix, stream=True)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"reference verification failed: {error}", file=sys.stderr)
        return 2
    if len(reference) != len(prefill) or len(prefill) != len(streaming):
        print("reference verification failed: hidden output widths differ", file=sys.stderr)
        return 2
    prefill_error = max(abs(expected - actual) for expected, actual in zip(reference, prefill))
    streaming_error = max(abs(expected - actual) for expected, actual in zip(reference, streaming))
    parity_error = max(abs(full - step) for full, step in zip(prefill, streaming))
    max_error = max(prefill_error, streaming_error, parity_error)
    report = {
        "schema_version": 1,
        "model": args.model_id,
        "revision": args.revision,
        "source_sha256": source_weight_digest(args.source),
        "converted_sha256": sha256(args.converted),
        "prefix": prefix,
        "hidden_values": len(reference),
        "full_prefix_max_absolute_error": prefill_error,
        "streaming_max_absolute_error": streaming_error,
        "full_prefix_vs_streaming_max_absolute_error": parity_error,
        "max_absolute_error": max_error,
        "tolerance": args.tolerance,
        "status": "passed" if max_error <= args.tolerance else "failed",
        "scope": (
            "real GPT-2 conversion plus full-prefix/streaming runtime parity; "
            "not policy inference"
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if max_error <= args.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
