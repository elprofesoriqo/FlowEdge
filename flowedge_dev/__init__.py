"""Maintainer library: bench, verify, convert, and rollout. Not the runtime wheel."""

__all__ = ["ROOT", "run"]

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(argv: list[str] | None = None) -> int:
    from .cli import main

    return main(argv)
