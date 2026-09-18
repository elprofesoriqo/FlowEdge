#!/usr/bin/env python3
"""Reference-machine Diffusion Policy rollout for Jetson, ARM, and x86 hosts.

Forwards to ``flowedge-lerobot-rollout`` with host facts. Joint-limit clamps
stay in the robot adapter; this script only drives the simulator seam.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from flowedge_lerobot.cli import main as rollout_main


def _has_flag(args: Sequence[str], name: str) -> bool:
    return name in args or any(item.startswith(f"{name}=") for item in args)


def _cuda_requested(args: Sequence[str]) -> bool:
    for index, item in enumerate(args):
        if item == "--device=cuda":
            return True
        if item == "--device" and index + 1 < len(args) and args[index + 1] == "cuda":
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    forwarded = list(args)
    if "--host-facts" not in forwarded:
        forwarded.append("--host-facts")
    if not _cuda_requested(forwarded) and not _has_flag(forwarded, "--threads"):
        forwarded.extend(["--threads", "4"])
    if not _has_flag(forwarded, "--period-ms"):
        forwarded.extend(["--period-ms", "10"])
    return rollout_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
