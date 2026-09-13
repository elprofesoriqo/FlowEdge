"""DeadlineFlow bridge for measured CPU Diffusion Policy plans.

The C++ selector can describe other hardware; this executor accepts only plans
it can actually execute. Quality scores must come from an external evaluation.
"""

import json
from math import isfinite
from pathlib import Path


class CpuPlanSelector:
    def __init__(self, plans, *, reserve_ms=1.0, minimum_quality=0.0):
        import flowedge

        if not isfinite(reserve_ms) or reserve_ms < 0:
            raise ValueError("reserve_ms must be finite and non-negative")
        if not isfinite(minimum_quality) or not 0 <= minimum_quality <= 1:
            raise ValueError("minimum_quality must be in [0, 1]")
        self.plans = {}
        native = []
        for item in plans:
            item = dict(item)
            if set(item) - {
                "id",
                "steps",
                "latency_ms",
                "quality",
                "memory_bytes",
                "solver",
                "precision",
                "trace_id",
            }:
                raise ValueError("unsupported CPU execution-plan fields")
            if item.get("precision", "fp32") != "fp32" or item.get("trace_id", 0) != 0:
                raise ValueError(
                    "CPU diffusion executor supports fp32 and no accelerator trace"
                )
            if item.get("solver", "ddim") != "ddim":
                raise ValueError(
                    "calibrated CPU plans currently require deterministic DDIM"
                )
            plan = flowedge.ExecutionPlan()
            plan.id = item["id"]
            plan.steps = item["steps"]
            plan.latency_ns = int(item["latency_ms"] * 1e6)
            plan.memory_bytes = item.get("memory_bytes", 0)
            plan.quality = item["quality"]
            native.append(plan)
            self.plans[plan.id] = item
        self.selector = flowedge.DeadlineFlow(native)
        self.reserve_ns = int(reserve_ms * 1e6)
        self.minimum_quality = minimum_quality

    @classmethod
    def from_json(cls, path, **kwargs):
        return cls(
            json.loads(Path(path).read_text(encoding="utf-8"))["plans"], **kwargs
        )

    def select(
        self,
        *,
        buffered_actions,
        period_ns,
        slack_ns,
        available_memory_bytes=2**64 - 1,
        healthy=True,
    ):
        import flowedge

        context = flowedge.PlanContext()
        context.buffered_actions = buffered_actions
        context.period_ns = period_ns
        context.slack_ns = slack_ns
        context.reserve_ns = self.reserve_ns
        context.minimum_quality = self.minimum_quality
        context.available_memory_bytes = available_memory_bytes
        context.healthy = healthy
        selected = self.selector.select(context)
        return None if selected is None else self.plans[selected]

    def observe(self, plan, latency_ns):
        if not self.selector.observe(plan["id"], max(1, latency_ns)):
            raise ValueError("unknown execution plan")
