#!/usr/bin/env python3
"""Construct a pinned upstream SmolVLA policy from local Hugging Face files.

This validates the source-policy loading boundary only. It neither calls
FlowEdge nor produces actions, so it is not a runtime-parity, latency, or
control-quality result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local lerobot/smolvla_base directory")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import lerobot
        import transformers
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(args.model)
        first_parameter = next(policy.parameters())
        report = {
            "schema_version": 1,
            "model": "lerobot/smolvla_base",
            "revision": args.revision,
            "policy_type": type(policy).__name__,
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "device": str(first_parameter.device),
            "lerobot_version": lerobot.__version__,
            "transformers_version": transformers.__version__,
            "status": "passed",
            "scope": (
                "upstream source-policy construction from local checkpoint and VLM weights; "
                "not FlowEdge inference, action generation, latency, or policy-quality evaluation"
            ),
        }
    except (ImportError, OSError, RuntimeError, StopIteration, ValueError) as error:
        print(f"SmolVLA source load failed: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
