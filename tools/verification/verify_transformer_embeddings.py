#!/usr/bin/env python3
"""Verify the external-embedding Transformer boundary on a real checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--module-path",
        type=Path,
        required=True,
        help="directory containing the built flowedge Python module",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_path))

    try:
        import numpy as np
        import flowedge
        from safetensors.numpy import load_file

        weights = load_file(str(args.checkpoint))
        token_weight = np.asarray(weights["transformer.embeddings.token.weight"], dtype=np.float32)
        tokens = np.asarray(args.tokens, dtype=np.int32)
        if tokens.size == 0 or np.any(tokens < 0) or np.any(tokens >= token_weight.shape[0]):
            raise ValueError("tokens must be non-empty and inside the checkpoint vocabulary")
        embeddings = np.ascontiguousarray(token_weight[tokens], dtype=np.float32)
        engine = flowedge.Engine(str(args.checkpoint), threads=0)
        token_hidden = engine.run(tokens)
        embedding_hidden = engine.run_embeddings(embeddings)
        into_hidden = np.empty_like(embedding_hidden)
        engine.run_embeddings_into(embeddings, into_hidden)
        embedding_error = float(np.max(np.abs(token_hidden - embedding_hidden)))
        into_error = float(np.max(np.abs(embedding_hidden - into_hidden)))
        if embedding_error > 5e-6 or into_error > 0.0:
            raise RuntimeError(
                f"embedding parity exceeded tolerance: token={embedding_error}, into={into_error}"
            )
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Transformer embedding verification failed: {error}", file=sys.stderr)
        return 2

    result = {
        "schema_version": 1,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "tokens": args.tokens,
        "hidden_values": int(token_hidden.size),
        "token_vs_embedding_max_absolute_error": embedding_error,
        "embedding_vs_into_max_absolute_error": into_error,
        "tolerance": 5e-6,
        "status": "passed",
        "scope": (
            "real converted GPT-2 token prefill versus caller-supplied embedding prefill; "
            "not SmolVLA or policy-quality evaluation"
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
