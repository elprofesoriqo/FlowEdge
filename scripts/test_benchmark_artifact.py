#!/usr/bin/env python3
"""Unit tests for the dependency-free benchmark artifact contract."""

import copy
import json
import unittest
from pathlib import Path

import benchmark_artifact


ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "bench" / "artifacts" / "lerobot-vs-flowedge.template.json"


def measured():
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    sample = {
        "status": "measured",
        "startup_ms": 3.0,
        "encoder_ms": {"p50": 1.0, "p95": 1.5, "p99": 2.0},
        "policy_ms": {"p50": 4.0, "p95": 5.0, "p99": 6.0},
        "end_to_end_ms": {"p50": 5.0, "p95": 6.5, "p99": 8.0},
        "throughput_hz": 200.0,
        "rss_mb": 64.0,
        "allocations": {"setup": 13, "hot_path": 0},
    }
    document["measurements"] = {"flowedge": sample, "lerobot": copy.deepcopy(sample)}
    return document


class BenchmarkArtifactTest(unittest.TestCase):
    def test_template_is_valid_and_explicitly_unmeasured(self):
        document = benchmark_artifact.validate(
            json.loads(TEMPLATE.read_text(encoding="utf-8"))
        )
        report = benchmark_artifact.render(document)
        self.assertIn("not measured", report)
        self.assertNotIn("x |", report)

    def test_measured_report_separates_encoder_and_policy(self):
        report = benchmark_artifact.render(benchmark_artifact.validate(measured()))
        self.assertIn("Encoder p50 / p95 / p99", report)
        self.assertIn("Policy p50 / p95 / p99", report)
        self.assertIn("Relative comparison", report)
        self.assertIn("13 / 0", report)

    def test_quantiles_must_be_monotonic(self):
        document = measured()
        document["measurements"]["flowedge"]["policy_ms"]["p95"] = 3.0
        with self.assertRaises(benchmark_artifact.ArtifactError):
            benchmark_artifact.validate(document)

    def test_missing_reference_is_allowed_but_marked(self):
        document = measured()
        document["measurements"]["lerobot"] = None
        report = benchmark_artifact.render(benchmark_artifact.validate(document))
        self.assertIn("| lerobot | not measured", report)
        self.assertNotIn("Relative comparison", report)


if __name__ == "__main__":
    unittest.main()
