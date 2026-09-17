#!/usr/bin/env python3
"""Print a kernel-mix table and the short-M GEMM roofline numbers.

Accepts the JSON from flowedge_diffusion_breakdown, a parquet/csv export, or
the committed example under data/samples/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Known mix lines that are a GEMM or an im2col GEMM. Bytes count A+W+C at f32.
GEMM_SHAPES: dict[str, tuple[int, int, int]] = {
    "gemm 4x10240x2048": (4, 10240, 2048),
    "gemm 4x10240x2048 t=2": (4, 10240, 2048),
    "gemm 4x10240x2048 t=4": (4, 10240, 2048),
    "conv 2048 L4 K5": (4, 10240, 2048),
    "conv 1024 L8 K5": (8, 5120, 1024),
    "conv 1024 L4 K5": (4, 5120, 1024),
    "conv 512 L16 K5": (16, 2560, 512),
    "conv 512 L8 K5": (8, 2560, 512),
    "conv 4096->1024 L4 K5": (4, 20480, 1024),
}


def arithmetic_intensity(m: int, k: int, n: int) -> dict[str, float]:
    flops = 2.0 * m * k * n
    bytes_ = 4.0 * (m * k + n * k + m * n)
    return {
        "flops": flops,
        "bytes": bytes_,
        "intensity": flops / bytes_,
        "weight_mb": (4.0 * n * k) / (1024.0 * 1024.0),
        "approx_m_over_2": m / 2.0,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload["rows"])
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pyarrow is required to read parquet") from exc
        table = pq.read_table(path)
        return table.to_pylist()
    raise SystemExit(f"unsupported mix file: {path}")


def as_float(value: Any) -> float:
    return float(value)


def as_int(value: Any) -> int:
    return int(float(value))


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    ordered = sorted(rows, key=lambda row: as_float(row["estimated_ms"]), reverse=True)
    print(title)
    print(f"{'kernel':<36} {'p50_us':>10} {'n':>5} {'ms/sample':>10} {'kind':<12} {'I':>7}")
    for row in ordered:
        name = str(row["name"])
        intensity = ""
        if name in GEMM_SHAPES:
            m, k, n = GEMM_SHAPES[name]
            intensity = f"{arithmetic_intensity(m, k, n)['intensity']:.2f}"
        print(
            f"{name:<36} {as_float(row['p50_us']):>10.1f} {as_int(row['calls']):>5} "
            f"{as_float(row['estimated_ms']):>10.1f} {str(row.get('kind', '')):<12} {intensity:>7}"
        )
    print()


def print_roofline() -> None:
    print("Short-M GEMM traffic (f32, A+W+C):")
    print(f"{'shape':<22} {'weight_MiB':>11} {'I flop/byte':>12} {'M/2':>6}")
    seen: set[tuple[int, int, int]] = set()
    for name, shape in GEMM_SHAPES.items():
        if not name.startswith("gemm ") or shape in seen:
            continue
        seen.add(shape)
        stats = arithmetic_intensity(*shape)
        label = f"{shape[0]}x{shape[1]}x{shape[2]}"
        print(
            f"{label:<22} {stats['weight_mb']:>11.1f} {stats['intensity']:>12.2f} "
            f"{stats['approx_m_over_2']:>6.1f}"
        )
    print()
    print("Ridge reminder: I ~ M/2 when W dominates. M=4 sits at 2 FLOP/byte.")
    print(
        f"70 calls x {arithmetic_intensity(4, 10240, 2048)['weight_mb']:.1f} MiB ~ "
        f"{70 * arithmetic_intensity(4, 10240, 2048)['weight_mb'] / 1024.0:.2f} GiB / sample."
    )
    print()


def compare(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    right_by_name = {str(row["name"]): row for row in right}
    print(f"{'kernel':<36} {'this_ms':>10} {'other_ms':>10} {'delta_ms':>10}")
    names = sorted(
        {str(row["name"]) for row in left} | set(right_by_name),
        key=lambda name: as_float(
            next((row["estimated_ms"] for row in left if str(row["name"]) == name), 0.0)
        ),
        reverse=True,
    )
    for name in names:
        this = next((row for row in left if str(row["name"]) == name), None)
        other = right_by_name.get(name)
        this_ms = as_float(this["estimated_ms"]) if this else math.nan
        other_ms = as_float(other["estimated_ms"]) if other else math.nan
        delta = this_ms - other_ms
        print(f"{name:<36} {this_ms:>10.1f} {other_ms:>10.1f} {delta:>10.1f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mix",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "samples" / "kernel_mix.example.json",
        help="json/parquet/csv from export_kernel_mix.py",
    )
    parser.add_argument("--vs", type=Path, help="optional second mix to diff against")
    args = parser.parse_args()
    rows = load_rows(args.mix)
    print_table(rows, f"mix: {args.mix}")
    print_roofline()
    if args.vs is not None:
        compare(rows, load_rows(args.vs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
