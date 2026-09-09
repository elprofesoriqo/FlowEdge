import unittest

import numpy as np

from flowedge_lerobot import FlowEdgeDiffusionPolicy, run_rollout


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
        result = run_rollout(policy, robot, lambda observation: observation, steps=3, seed=7)

        self.assertEqual(result.steps, 3)
        self.assertTrue(result.stopped)
        self.assertEqual(robot.reset_count, 1)
        self.assertEqual(robot.stop_count, 1)
        self.assertEqual(len(robot.actions), 3)
        self.assertEqual(robot.actions[0].shape, (2,))

    def test_stop_runs_when_encoding_fails(self):
        robot = FakeRobot()
        policy = FlowEdgeDiffusionPolicy(FakeEngine())

        def fail(_):
            raise RuntimeError("encoder failure")

        with self.assertRaisesRegex(RuntimeError, "encoder failure"):
            run_rollout(policy, robot, fail, steps=1)
        self.assertEqual(robot.stop_count, 1)

    def test_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "steps must be positive"):
            run_rollout(
                FlowEdgeDiffusionPolicy(FakeEngine()),
                FakeRobot(),
                lambda value: value,
                steps=0,
            )


if __name__ == "__main__":
    unittest.main()
