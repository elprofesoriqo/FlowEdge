#!/usr/bin/env python3
"""Compare Google Benchmark JSON artifacts from one host.

The script deliberately compares CPU time, not wall time, and only consumes the
``mean`` aggregate emitted by ``--benchmark_report_aggregates_only``.  Keeping
capture and comparison separate makes artifacts portable and lets a reviewer
rerun the comparison without rebuilding either revision.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TIME_TO_NS = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}


@dataclass(frozen=True)
class Measurement:
    name: str
    cpu_ns: float
    iterations: int


def read_measurements(path: Path) -> tuple[dict[str, Any], dict[str, Measurement]]:
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, dict) or not isinstance(document.get("benchmarks"), list):
        raise ValueError(f"{path} is not Google Benchmark JSON")

    measurements: dict[str, Measurement] = {}
    for row in document["benchmarks"]:
        if not isinstance(row, dict):
            continue
        if row.get("run_type") != "aggregate" or row.get("aggregate_name") != "mean":
            continue
        run_name = row.get("run_name")
        cpu_time = row.get("cpu_time")
        if not isinstance(run_name, str) or not isinstance(cpu_time, (int, float)):
            continue
        unit = row.get("time_unit", document.get("time_unit", "ns"))
        if unit not in TIME_TO_NS:
            raise ValueError(f"{path}: unsupported time unit {unit!r} for {run_name}")
        measurements[run_name] = Measurement(
            run_name, float(cpu_time) * TIME_TO_NS[unit], int(row.get("iterations", 0))
        )
    if not measurements:
        raise ValueError(
            f"{path} has no mean aggregates; capture with --benchmark_repetitions and "
            "--benchmark_report_aggregates_only=true"
        )
    return document.get("context", {}), measurements


def context_line(context: dict[str, Any]) -> str:
    values = [
        str(context.get("host_name", platform.node())),
        f"{context.get('num_cpus', '?')} logical CPUs @ {context.get('mhz_per_cpu', '?')} MHz",
        str(context.get("library_build_type", "unknown build")),
    ]
    return " — ".join(values)


def render_report(
    baseline_path: Path,
    candidate_path: Path,
    threshold: float,
    baseline_context: dict[str, Any],
    candidate_context: dict[str, Any],
    baseline: dict[str, Measurement],
    candidate: dict[str, Measurement],
    allowed_removed: list[re.Pattern[str]],
) -> tuple[str, bool]:
    shared = sorted(set(baseline) & set(candidate))
    removed = sorted(set(baseline) - set(candidate))
    missing_from_candidate = [
        name for name in removed if not any(pattern.search(name) for pattern in allowed_removed)
    ]
    allowed_from_candidate = [name for name in removed if name not in missing_from_candidate]
    missing_from_baseline = sorted(set(candidate) - set(baseline))
    regressed = False
    lines = [
        "# Kernel benchmark comparison",
        "",
        f"- Baseline: `{baseline_path}` ({context_line(baseline_context)})",
        f"- Candidate: `{candidate_path}` ({context_line(candidate_context)})",
        f"- Regression budget: {threshold:.2f}% CPU time",
        "",
        "| Benchmark | Baseline | Candidate | Change | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for name in shared:
        before, after = baseline[name].cpu_ns, candidate[name].cpu_ns
        change = ((after / before) - 1.0) * 100.0 if before else float("inf")
        status = "REGRESSION" if change > threshold else "ok"
        regressed |= status == "REGRESSION"
        lines.append(
            f"| `{name}` | {before / 1_000.0:.3f} us | {after / 1_000.0:.3f} us | "
            f"{change:+.2f}% | {status} |"
        )
    if not shared:
        lines.extend(["", "No benchmark names matched; the comparison is invalid."])
        regressed = True
    if missing_from_candidate:
        lines.extend(["", "## Missing from candidate", "", *[f"- `{name}`" for name in missing_from_candidate]])
        regressed = True
    if allowed_from_candidate:
        lines.extend(
            [
                "",
                "## Intentionally retired or fused",
                "",
                *[f"- `{name}`" for name in allowed_from_candidate],
            ]
        )
    if missing_from_baseline:
        lines.extend(["", "## New in candidate", "", *[f"- `{name}`" for name in missing_from_baseline]])
    lines.extend(
        [
            "",
            "A result above budget is a review signal. Re-run on an idle, fixed-power host before "
            "attributing a small change to code.",
        ]
    )
    return "\n".join(lines) + "\n", regressed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline Google Benchmark JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate Google Benchmark JSON")
    parser.add_argument("--threshold", type=float, default=5.0, help="allowed CPU-time increase in percent")
    parser.add_argument("--report", type=Path, help="write Markdown report to this path")
    parser.add_argument("--no-fail", action="store_true", help="always exit zero after writing the report")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="map a renamed baseline benchmark to its candidate name (repeatable)",
    )
    parser.add_argument(
        "--allow-removed",
        action="append",
        default=[],
        metavar="REGEX",
        help="allow a removed baseline benchmark, for example after kernel fusion",
    )
    args = parser.parse_args()
    if args.threshold < 0:
        parser.error("--threshold must be non-negative")
    try:
        base_context, baseline = read_measurements(args.baseline)
        candidate_context, candidate = read_measurements(args.candidate)
        for mapping in args.rename:
            old, separator, new = mapping.partition("=")
            if not separator or not old or not new:
                raise ValueError(f"invalid --rename {mapping!r}; expected OLD=NEW")
            if old not in baseline:
                raise ValueError(f"--rename baseline name not found: {old}")
            if new in baseline and new != old:
                raise ValueError(f"--rename target already exists in baseline: {new}")
            measurement = baseline.pop(old)
            baseline[new] = Measurement(new, measurement.cpu_ns, measurement.iterations)
        allowed_removed = [re.compile(pattern) for pattern in args.allow_removed]
    except (OSError, ValueError, re.error, json.JSONDecodeError) as error:
        print(f"benchmark comparison failed: {error}", file=sys.stderr)
        return 2
    report, regressed = render_report(
        args.baseline,
        args.candidate,
        args.threshold,
        base_context,
        candidate_context,
        baseline,
        candidate,
        allowed_removed,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if args.no_fail or not regressed else 1


if __name__ == "__main__":
    raise SystemExit(main())
