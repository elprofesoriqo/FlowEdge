import importlib.util
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("lerobot") and importlib.util.find_spec("torchvision"),
    "requires the full optional LeRobot runtime",
)
class PluginTests(unittest.TestCase):
    def make_policy(self, seed=7):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig
        from lerobot_policy_flowedge.modeling_flowedge import FlowEdgePolicy
        from flowedge_lerobot import FlowEdgeDiffusionPolicy

        class Engine:
            diffusion_metadata = {
                "condition_dim": 3,
                "action_dim": 2,
                "horizon": 4,
                "action_steps": 3,
                "observation_steps": 2,
            }

            def __init__(self):
                self.noises = []

            def sample_diffusion(self, condition, noise, steps, scheduler, seed):
                self.noises.append(noise.copy())
                return np.array(
                    [[0, 1], [10, 11], [20, 21], [30, 31]], dtype=np.float32
                )

        engine = Engine()
        config = FlowEdgeConfig(
            checkpoint_path="fixture",
            seed=seed,
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
            },
        )
        with patch(
            "flowedge_lerobot.FlowEdgeDiffusionPolicy.from_checkpoint",
            return_value=FlowEdgeDiffusionPolicy(engine),
        ):
            return FlowEdgePolicy(config), engine

    def test_select_consumes_chunk_then_refills(self):
        import torch

        policy, engine = self.make_policy()
        batch = {"observation.state": torch.zeros(1, 3)}
        actions = [policy.select_action(batch).tolist() for _ in range(4)]
        self.assertEqual(actions, [[[10, 11]], [[20, 21]], [[30, 31]], [[10, 11]]])
        self.assertEqual(len(engine.noises), 2)
        self.assertFalse(np.array_equal(engine.noises[0], engine.noises[1]))
        self.assertTrue(engine.noises[0].any())

    def test_reset_replays_seed_and_discards_pending_actions(self):
        import torch

        policy, engine = self.make_policy()
        batch = {"observation.state": torch.zeros(1, 3)}
        original = policy.select_action(batch)
        policy.select_action(batch)
        policy.reset()
        self.assertTrue(torch.equal(original, policy.select_action(batch)))
        np.testing.assert_array_equal(engine.noises[0], engine.noises[1])

    def test_explicit_noise_and_direct_prediction_do_not_reuse_stale_queue(self):
        import torch

        policy, engine = self.make_policy()
        batch = {"observation.state": torch.zeros(1, 3)}
        policy.select_action(batch)
        policy.predict_action_chunk(batch, noise=torch.ones(1, 4, 2))
        np.testing.assert_array_equal(engine.noises[-1], np.ones((4, 2)))
        self.assertEqual(policy.select_action(batch).tolist(), [[10, 11]])
        self.assertEqual(len(engine.noises), 3)

    def test_invalid_conditions_and_noise_fail(self):
        import torch

        policy, _ = self.make_policy()
        with self.assertRaises(ValueError):
            policy.select_action({"observation.state": torch.zeros(1, 2)})
        with self.assertRaises(ValueError):
            policy.predict_action_chunk(
                {"observation.state": torch.zeros(1, 3)}, noise=np.zeros((2, 2))
            )

    def test_encoded_processor_rejects_raw_dataset_stats(self):
        from lerobot_policy_flowedge.processor_flowedge import (
            make_flowedge_pre_post_processors,
        )

        policy, _ = self.make_policy()
        with self.assertRaisesRegex(ValueError, "already be normalized"):
            make_flowedge_pre_post_processors(
                policy.config, {"observation.state": {"min": 0}}
            )

    def test_plugin_registers_flowedge_policy(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        self.assertIsNotNone(PreTrainedConfig.get_choice_class("flowedge"))
        self.assertIsNotNone(PreTrainedConfig.get_choice_class("flowedge_smolvla"))

    def test_smolvla_plugin_consumes_cached_chunks(self):
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeSmolVLAConfig
        from lerobot_policy_flowedge.modeling_flowedge_smolvla import FlowEdgeSmolVLAPolicy
        from flowedge_lerobot import FlowEdgeSmolVLACachedExpert, SmolVLAKVCache

        class Engine:
            model_metadata = {
                "architecture": 7,
                "action_dim": 2,
                "action_horizon": 3,
                "n_layers": 2,
            }

            def smolvla_sample(self, noise, prefix_keys, prefix_values, prefix_mask, steps):
                del noise, prefix_keys, prefix_values, prefix_mask, steps
                return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        class Provider:
            def reset(self):
                self.resets = getattr(self, "resets", 0) + 1

            def __call__(self, batch):
                del batch
                return SmolVLAKVCache(
                    keys=np.zeros((2, 3, 320), dtype=np.float32),
                    values=np.zeros((2, 3, 320), dtype=np.float32),
                    mask=np.array([1, 1, 0], dtype=np.uint8),
                )

        config = FlowEdgeSmolVLAConfig(
            checkpoint_path="expert.safetensors",
            source_checkpoint_path="smolvla_base",
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
            },
        )
        policy = FlowEdgeSmolVLAPolicy(
            config,
            expert=FlowEdgeSmolVLACachedExpert(Engine(), action_dim=2, action_steps=2),
            cache_provider=Provider(),
        )
        batch = {"observation.state": torch.zeros(1, 6)}
        self.assertEqual(policy.select_action(batch).tolist(), [[1, 2]])
        self.assertEqual(policy.select_action(batch).tolist(), [[3, 4]])
        self.assertEqual(policy.select_action(batch).tolist(), [[1, 2]])


if __name__ == "__main__":
    unittest.main()
