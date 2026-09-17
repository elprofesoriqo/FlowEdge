#!/usr/bin/env python3
"""Reference-machine Diffusion Policy rollout for Jetson, ARM, and x86 hosts.

Forwards to ``flowedge-lerobot-rollout`` with host facts. Joint-limit clamps
stay in the robot adapter; this script only drives the simulator seam.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from flowedge_lerobot.cli import main as rollout_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    forwarded = list(args)
    if "--host-facts" not in forwarded:
        forwarded.append("--host-facts")
    if "--threads" not in forwarded and not any(item.startswith("--threads=") for item in forwarded):
        forwarded.extend(["--threads", "4"])
    if "--period-ms" not in forwarded and not any(
        item.startswith("--period-ms=") for item in forwarded
    ):
        forwarded.extend(["--period-ms", "10"])
    return rollout_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
