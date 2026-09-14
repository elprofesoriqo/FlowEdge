#!/usr/bin/env python3
"""Compare a converted real GPT-2 checkpoint with Hugging Face on one prefix.

This is a conversion/runtime smoke check, not a policy-quality or latency
benchmark. It intentionally uses a downloaded checkpoint instead of generated
weights and checks the native FlowEdge executable against the source model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from transformers import GPT2Model


DEFAULT_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local GPT-2-style source directory")
    parser.add_argument("converted", type=Path, help="FlowEdge converted safetensors file")
    parser.add_argument(
        "--binary", type=Path, required=True, help="built transformer_forward executable"
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tolerance", type=float, default=5e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    try:
        model = GPT2Model.from_pretrained(args.source).eval()
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            reference = float(model(input_ids=tokens).last_hidden_state[0, 0, 0])
        completed = subprocess.run(
            [str(args.binary), runtime_path(args.converted, args.binary), "1", "2", "3", "4"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"reference verification failed: {error}", file=sys.stderr)
        return 2
    match = re.search(r"out\[0\]=([-+0-9.eE]+)", completed.stdout)
    if completed.returncode != 0 or match is None:
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        print("reference verification failed: FlowEdge prefix execution failed", file=sys.stderr)
        return 2
    flowedge = float(match.group(1))
    absolute_error = abs(reference - flowedge)
    report = {
        "schema_version": 1,
        "model": "sshleifer/tiny-gpt2",
        "revision": args.revision,
        "source_sha256": sha256(args.source / "pytorch_model.bin"),
        "converted_sha256": sha256(args.converted),
        "prefix": [1, 2, 3, 4],
        "reference_first_hidden": reference,
        "flowedge_first_hidden": flowedge,
        "absolute_error": absolute_error,
        "tolerance": args.tolerance,
        "status": "passed" if absolute_error <= args.tolerance else "failed",
        "scope": "real GPT-2 conversion and fixed-prefix runtime smoke check; not policy inference",
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if absolute_error <= args.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
