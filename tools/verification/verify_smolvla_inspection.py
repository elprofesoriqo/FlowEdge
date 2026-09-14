#!/usr/bin/env python3
"""Verify native recognition of the pinned real SmolVLA checkpoint.

This is deliberately an inspection/provenance check.  A passing report means
FlowEdge recognizes the source tensor schema and rejects it as unsupported; it
does not mean that FlowEdge executes SmolVLA inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PINNED_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        completed = subprocess.run(
            [str(args.binary), str(args.checkpoint), "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
        report: dict[str, Any] = json.loads(completed.stdout)
        if completed.returncode != 2:
            raise RuntimeError(
                f"inspector must reject unsupported SmolVLA with exit code 2, got {completed.returncode}"
            )
        model = report["model"]
        checkpoint = report["checkpoint"]
        compatibility = report["compatibility"]
        errors = compatibility["errors"]
        required_errors = {
            "SmolVLA full VLM encoder and interleaved action-expert attention are not implemented",
            "SmolVLA requires the LeRobot image/language preprocessing and observation-history boundary",
        }
        if model["family"] != "smolvla":
            raise RuntimeError(f"unexpected model family: {model['family']!r}")
        if model["d_model"] != 960 or model["d_inner"] != 720 or model["layers"] != 16:
            raise RuntimeError("unexpected SmolVLA expert dimensions")
        if checkpoint["tensor_count"] != 500:
            raise RuntimeError("unexpected SmolVLA tensor count")
        if checkpoint["precision"] != "mixed":
            raise RuntimeError("unexpected SmolVLA precision classification")
        if report["memory"]["arena_bytes"] <= 0:
            raise RuntimeError("SmolVLA suffix boundary must reserve a runtime arena")
        if compatibility["supported"] or compatibility["missing_required_tensors"]:
            raise RuntimeError("inspector admitted or incompletely recognized SmolVLA")
        if not required_errors.issubset(errors):
            raise RuntimeError("inspector omitted an explicit unsupported-boundary reason")
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"SmolVLA inspection verification failed: {error}", file=sys.stderr)
        return 2

    result = {
        "schema_version": 1,
        "model": "lerobot/smolvla_base",
        "revision": PINNED_REVISION,
        "checkpoint": args.checkpoint.name,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(args.checkpoint),
        "native_digest": checkpoint["digest"],
        "family": model["family"],
        "expert_width": model["d_inner"],
        "expert_layers": model["layers"],
        "tensor_count": checkpoint["tensor_count"],
        "precision": checkpoint["precision"],
        "arena_bytes": report["memory"]["arena_bytes"],
        "supported_by_flowedge": compatibility["supported"],
        "status": "passed",
        "scope": (
            "real SmolVLA safetensors schema recognition and explicit full-policy boundary; "
            "suffix projection parity is covered separately; no action generation, latency, "
            "or policy-quality evaluation"
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
