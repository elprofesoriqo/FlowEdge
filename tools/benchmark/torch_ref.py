#!/usr/bin/env python3
"""PyTorch reference utility for FlowEdge's Mamba verification checks."""

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    mode = args[0] if args else "dump"
    if mode != "dump":
        raise SystemExit("usage: torch_ref.py [dump] [MODEL] [OUTPUT_PREFIX]")
    from torch_mamba_reference import MambaReference

    reference = MambaReference(args[1] if len(args) > 1 else "models/mamba.safetensors")
    reference.dump(args[2] if len(args) > 2 else "models/baseline")


if __name__ == "__main__":
    main()
