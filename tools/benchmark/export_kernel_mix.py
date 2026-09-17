#!/usr/bin/env python3
"""Run flowedge_diffusion_breakdown and save the mix under data/.

The C++ tool already multiplies per-shape p50 by the 10-step DDIM call count.
This wrapper exists so those rows land in parquet/csv next to the committed
sample, which is what data/explore.py reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def find_binary(build_dir: Path) -> Path:
    names = ("flowedge_diffusion_breakdown.exe", "flowedge_diffusion_breakdown")
    for name in names:
        candidate = build_dir / name
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"no breakdown binary in {build_dir}. Build flowedge_diffusion_breakdown "
        "with FLOWEDGE_BENCH=ON."
    )


def write_tabular(rows: list[dict], json_path: Path) -> None:
    csv_path = json_path.with_suffix(".csv")
    fieldnames = ["name", "p50_us", "calls", "estimated_ms", "kind"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"wrote {csv_path}")

    parquet_path = json_path.with_suffix(".parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow not installed; skipped parquet (csv/json are enough for explore.py)")
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)
    print(f"wrote {parquet_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.environ.get("FLOWEDGE_BUILD_DIR", ROOT / "build-win-clang")),
    )
    parser.add_argument("--checkpoint", type=Path, help="optional converted .safetensors")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "kernel_mix.json",
        help="JSON path; csv/parquet share the same stem",
    )
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="do not run the binary; convert --output json to csv/parquet",
    )
    args = parser.parse_args()

    if not args.from_json:
        binary = find_binary(args.build_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(binary), str(args.output)]
        if args.checkpoint is not None:
            command.insert(1, str(args.checkpoint))
        print(" ".join(command))
        subprocess.run(command, check=True)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    write_tabular(payload["rows"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
