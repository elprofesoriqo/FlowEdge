import unittest

import numpy as np

from flowedge_lerobot import FlowEdgeDiffusionPolicy, MAX_ROLLOUT_STEPS, run_rollout
from flowedge_lerobot.rollout import DeadlineMissed


class FakeEngine:
    diffusion_metadata = {
        "condition_dim": 3,
        "action_dim": 2,
        "horizon": 4,
        "action_steps": 2,
        "observation_steps": 1,
    }

    def sample_diffusion(self, condition, noise, steps, scheduler, seed):
        del condition, steps, scheduler, seed
        return noise


class FakeRobot:
    def __init__(self):
        self.observations = 0
        self.actions = []
        self.reset_count = 0
        self.stop_count = 0

    def reset(self):
        self.reset_count += 1

    def observe(self):
        self.observations += 1
        return np.full(3, self.observations, dtype=np.float32)

    def send_action(self, action):
        self.actions.append(action)

    def stop(self):
        self.stop_count += 1


class RolloutTests(unittest.TestCase):
    def test_bounded_loop_stops(self):
        robot = FakeRobot()
        policy = FlowEdgeDiffusionPolicy(FakeEngine())
        result = run_rollout(
            policy, robot, lambda observation: observation, steps=3, seed=7
        )

        self.assertEqual(result.steps, 3)
        self.assertTrue(result.stopped)
        self.assertEqual(robot.reset_count, 1)
        self.assertEqual(robot.stop_count, 1)
        self.assertEqual(len(robot.actions), 3)
        self.assertEqual(robot.actions[0].shape, (2,))
        self.assertGreater(result.p99_ms, 0.0)
        self.assertGreaterEqual(result.p99_ms, result.p50_ms)
        self.assertEqual(result.missed_deadlines, 0)

    def test_stop_runs_when_encoding_fails(self):
        robot = FakeRobot()
        policy = FlowEdgeDiffusionPolicy(FakeEngine())

        def fail(_):
            raise RuntimeError("encoder failure")

        with self.assertRaisesRegex(RuntimeError, "encoder failure"):
            run_rollout(policy, robot, fail, steps=1)
        self.assertEqual(robot.stop_count, 1)

    def test_stop_runs_when_robot_reset_fails(self):
        robot = FakeRobot()
        policy = FlowEdgeDiffusionPolicy(FakeEngine())

        def fail_reset():
            raise RuntimeError("reset failure")

        robot.reset = fail_reset
        with self.assertRaisesRegex(RuntimeError, "reset failure"):
            run_rollout(policy, robot, lambda value: value, steps=1)
        self.assertEqual(robot.stop_count, 1)

    def test_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "steps must be from 1"):
            run_rollout(
                FlowEdgeDiffusionPolicy(FakeEngine()),
                FakeRobot(),
                lambda value: value,
                steps=0,
            )

    def test_period_counts_missed_steps(self):
        result = run_rollout(
            FlowEdgeDiffusionPolicy(FakeEngine()),
            FakeRobot(),
            lambda value: value,
            steps=2,
            period_ms=0.001,
        )
        self.assertEqual(result.missed_deadlines, 2)
        self.assertEqual(result.on_miss, "hold")

    def test_on_miss_drop_skips_send(self):
        robot = FakeRobot()
        run_rollout(
            FlowEdgeDiffusionPolicy(FakeEngine()),
            robot,
            lambda value: value,
            steps=2,
            period_ms=0.001,
            on_miss="drop",
        )
        self.assertEqual(robot.actions, [])

    def test_on_miss_hold_repeats_last_sent_action(self):
        robot = FakeRobot()
        run_rollout(
            FlowEdgeDiffusionPolicy(FakeEngine()),
            robot,
            lambda value: value,
            steps=2,
            period_ms=0.001,
            on_miss="hold",
        )
        self.assertEqual(len(robot.actions), 2)
        np.testing.assert_array_equal(robot.actions[0], np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(robot.actions[1], np.zeros(2, dtype=np.float32))

    def test_on_miss_raise(self):
        robot = FakeRobot()
        with self.assertRaises(DeadlineMissed):
            run_rollout(
                FlowEdgeDiffusionPolicy(FakeEngine()),
                robot,
                lambda value: value,
                steps=1,
                period_ms=0.001,
                on_miss="raise",
            )
        self.assertEqual(robot.stop_count, 1)

    def test_on_miss_must_be_known(self):
        with self.assertRaisesRegex(ValueError, "on_miss must be"):
            run_rollout(
                FlowEdgeDiffusionPolicy(FakeEngine()),
                FakeRobot(),
                lambda value: value,
                steps=1,
                on_miss="zero",
            )

    def test_period_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "period_ms must be positive"):
            run_rollout(
                FlowEdgeDiffusionPolicy(FakeEngine()),
                FakeRobot(),
                lambda value: value,
                steps=1,
                period_ms=0,
            )

    def test_rollout_step_bound_and_finite_period(self):
        policy = FlowEdgeDiffusionPolicy(FakeEngine())
        with self.assertRaisesRegex(ValueError, "steps must be from 1"):
            run_rollout(
                policy, FakeRobot(), lambda value: value, steps=MAX_ROLLOUT_STEPS + 1
            )
        with self.assertRaisesRegex(ValueError, "period_ms must be positive"):
            run_rollout(
                policy,
                FakeRobot(),
                lambda value: value,
                steps=1,
                period_ms=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
