#!/usr/bin/env python3
"""PyTorch references for FlowEdge's checkpoint and latency benchmarks."""

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    mode = args[0] if args else "dump"
    if mode == "latency":
        from torch_latency import run

        run(args)
        return
    from torch_mamba_reference import MambaReference

    reference = MambaReference(args[1] if len(args) > 1 else "models/mamba.safetensors")
    if mode == "bench":
        reference.bench(int(args[2]) if len(args) > 2 else 1)
    else:
        reference.dump(args[2] if len(args) > 2 else "models/baseline")


if __name__ == "__main__":
    main()
