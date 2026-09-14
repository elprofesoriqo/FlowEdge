import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("flowedge"), "requires native extension")
class DeadlineTests(unittest.TestCase):
    def test_cpu_selection_and_feedback_use_native_selector(self):
        from flowedge_lerobot.deadline import CpuPlanSelector

        selector = CpuPlanSelector(
            [
                {"id": 1, "steps": 4, "latency_ms": 20, "quality": 0.5},
                {"id": 2, "steps": 10, "latency_ms": 60, "quality": 0.9},
            ],
            reserve_ms=0,
        )
        args = dict(buffered_actions=5, period_ns=20_000_000, slack_ns=100_000_000)
        self.assertEqual(selector.select(**args)["id"], 2)
        selector.observe(selector.plans[2], 110_000_000)
        self.assertEqual(selector.select(**args)["id"], 1)
        self.assertIsNone(selector.select(**args, healthy=False))

    def test_executor_rejects_unimplemented_plans(self):
        from flowedge_lerobot.deadline import CpuPlanSelector

        for change in ({"precision": "bf16"}, {"trace_id": 3}, {"solver": "euler"}):
            with self.assertRaises(ValueError):
                CpuPlanSelector(
                    [{"id": 1, "steps": 4, "latency_ms": 20, "quality": 0.5, **change}]
                )

    def test_invalid_and_duplicate_plans_fail_setup(self):
        from flowedge_lerobot.deadline import CpuPlanSelector

        plan = {"id": 1, "steps": 4, "latency_ms": 20, "quality": 0.5}
        for plans in (
            [],
            [plan, plan],
            [{**plan, "quality": float("nan")}],
            [{**plan, "steps": 0}],
        ):
            with self.assertRaises(ValueError):
                CpuPlanSelector(plans)
