#!/usr/bin/env python3
"""Report checked-in binary-size and setup-allocation budgets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def find_library(build_dir: Path, names: tuple[str, ...]) -> Path:
    candidates = [path for name in names for path in build_dir.rglob(name)]
    candidates = [path for path in candidates if "_deps" not in path.parts]
    if not candidates:
        raise FileNotFoundError(f"could not find {', '.join(names)} below {build_dir}")
    return max(candidates, key=lambda path: path.stat().st_size)


def setup_allocations(executable: Path, model: Path) -> int:
    completed = subprocess.run(
        [str(executable), str(model), str(model), "1"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    match = re.search(
        r"^FP32\s+\|.*\|\s*(\d+)\s+\|\s*(\d+)\s*$", completed.stdout, re.MULTILINE
    )
    if match is None:
        raise RuntimeError("model latency benchmark did not report an FP32 allocation row")
    hot_allocations = int(match.group(2))
    if hot_allocations != 0:
        raise RuntimeError(f"model hot path allocated {hot_allocations} times")
    return int(match.group(1))


def render(measured: dict[str, int], budgets: dict[str, int]) -> tuple[str, bool]:
    rows = [
        "# FlowEdge size and setup budget",
        "",
        "| Metric | Measured | Budget | Result |",
        "|---|---:|---:|---|",
    ]
    failed = False
    for name, label in (
        ("core_bytes", "Core library bytes"),
        ("relay_bytes", "Relay library bytes"),
        ("model_setup_allocations", "Model setup allocations"),
    ):
        actual = measured[name]
        budget = budgets[name]
        result = "ok" if actual <= budget else "OVER BUDGET"
        failed |= result != "ok"
        rows.append(f"| {label} | {actual} | {budget} | {result} |")
    rows.extend(
        [
            "",
            "The setup allocation count covers engine construction; the benchmark also fails if the "
            "measured model hot path allocates.",
        ]
    )
    return "\n".join(rows) + "\n", failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--budget", type=Path, default=Path("bench/budgets.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        budget_document = json.loads(args.budget.read_text(encoding="utf-8"))
        if budget_document.get("version") != 1:
            raise ValueError("unsupported budget version")
        max_library = budget_document["max_library_bytes"]
        budgets = {
            "core_bytes": int(max_library["core"]),
            "relay_bytes": int(max_library["relay"]),
            "model_setup_allocations": int(budget_document["max_model_setup_allocations"]),
        }
        core = find_library(args.build_dir, ("libflowedge_engine.a", "flowedge_engine.lib"))
        relay = find_library(args.build_dir, ("libflowedge_relay.a", "flowedge_relay.lib"))
        benchmark = next(
            (
                path
                for path in (
                    args.build_dir / "flowedge_model_latency_bench",
                    args.build_dir / "flowedge_model_latency_bench.exe",
                    args.build_dir / "Release" / "flowedge_model_latency_bench.exe",
                )
                if path.is_file()
            ),
            None,
        )
        if benchmark is None:
            raise FileNotFoundError("flowedge_model_latency_bench was not found")
        measured = {
            "core_bytes": core.stat().st_size,
            "relay_bytes": relay.stat().st_size,
            "model_setup_allocations": setup_allocations(benchmark, args.model),
        }
        report, failed = render(measured, budgets)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"budget report failed: {error}", file=sys.stderr)
        return 2

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
