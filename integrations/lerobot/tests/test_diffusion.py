import numpy as np
import unittest

from flowedge_lerobot import DiffusionActionContract, FlowEdgeDiffusionPolicy


class FakeEngine:
    diffusion_metadata = {
        "condition_dim": 4,
        "action_dim": 2,
        "horizon": 6,
        "action_steps": 3,
        "observation_steps": 2,
    }

    def __init__(self):
        self.calls = []

    def sample_diffusion(self, condition, noise, steps, scheduler, seed):
        self.calls.append((condition, noise, steps, scheduler, seed))
        return np.arange(12, dtype=np.float32).reshape(6, 2)


class DiffusionAdapterTests(unittest.TestCase):
    def test_contract_slices_the_le_robot_action_chunk(self):
        contract = DiffusionActionContract.from_metadata(FakeEngine.diffusion_metadata)
        self.assertEqual(contract.action_start, 1)
        self.assertEqual(contract.action_steps, 3)


    def test_adapter_forwards_contiguous_inputs_and_slices_horizon(self):
        engine = FakeEngine()
        policy = FlowEdgeDiffusionPolicy(engine)
        condition = np.arange(4, dtype=np.float64).reshape(1, 4)
        noise = np.zeros((6, 2), dtype=np.float64)

        actions = policy.predict_action_chunk(condition, noise, steps=7, seed=11)

        np.testing.assert_array_equal(actions, [[2, 3], [4, 5], [6, 7]])
        self.assertEqual(engine.calls[0][0].dtype, np.float32)
        self.assertTrue(engine.calls[0][0].flags.c_contiguous)
        self.assertEqual(engine.calls[0][1].dtype, np.float32)
        self.assertTrue(engine.calls[0][1].flags.c_contiguous)
        self.assertEqual(engine.calls[0][2:], (7, "ddim", 11))


    def test_adapter_rejects_shape_mismatch(self):
        cases = (
            (np.zeros(3), np.zeros((6, 2)), "condition"),
            (np.zeros(4), np.zeros((5, 2)), "noise"),
        )
        for condition, noise, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    FlowEdgeDiffusionPolicy(FakeEngine()).predict_action_chunk(condition, noise)
