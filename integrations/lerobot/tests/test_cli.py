import json
import unittest
from unittest.mock import patch

from flowedge_lerobot.cli import _parser, main, run_simulator


class FakePolicy:
    metadata = type("Metadata", (), {"condition_dim": 3, "horizon": 4})()
    action_dim = 2

    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def select_action_into(self, condition, noise, output, **kwargs):
        self.calls += 1
        output.fill(float(condition[0] + noise[0, 0] + kwargs["seed"]))


class CliTests(unittest.TestCase):
    def test_parser_defaults(self):
        args = _parser().parse_args(["policy.safetensors"])
        self.assertEqual(args.steps, 10)
        self.assertEqual(args.diffusion_steps, 10)
        self.assertEqual(args.scheduler, "ddim")
        self.assertIsNone(args.period_ms)

    def test_simulator_is_bounded_and_stops(self):
        policy = FakePolicy()
        result = run_simulator(
            policy, steps=3, diffusion_steps=2, scheduler="ddim", seed=7
        )
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.stopped)
        self.assertEqual(policy.calls, 3)

    def test_main_loads_checkpoint_and_emits_result(self):
        with patch(
            "flowedge_lerobot.cli.FlowEdgeDiffusionPolicy.from_checkpoint"
        ) as load:
            load.return_value = FakePolicy()
            with patch("builtins.print") as print_result:
                self.assertEqual(main(["policy.safetensors", "--steps", "2"]), 0)
        load.assert_called_once_with("policy.safetensors", threads=None)
        payload = json.loads(print_result.call_args.args[0])
        self.assertEqual(payload["steps"], 2)
        self.assertTrue(payload["stopped"])
        self.assertIn("p99_ms", payload)
        self.assertGreaterEqual(payload["startup_ms"], 0.0)
        self.assertTrue(payload["platform"])
        self.assertTrue(payload["machine"])


if __name__ == "__main__":
    unittest.main()
