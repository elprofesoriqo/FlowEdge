import importlib.util
import unittest

import numpy as np

from flowedge_lerobot import LeRobotSmolVLACacheProvider


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("lerobot"),
    "requires torch and LeRobot",
)
class SmolVLASourceProviderTests(unittest.TestCase):
    def test_provider_replays_source_prefix_and_flattens_cache(self):
        import torch

        testcase = self

        class VlmWithExpert:
            def forward(self, **kwargs):
                testcase.assertTrue(
                    torch.equal(
                        kwargs["attention_mask"],
                        torch.tensor(
                            [[[True, False, True], [False, False, False], [True, False, True]]]
                        ),
                    )
                )
                cache = {
                    layer: {
                        "key_states": torch.full((1, 3, 5, 64), float(layer)),
                        "value_states": torch.full((1, 3, 5, 64), float(layer + 1)),
                    }
                    for layer in range(2)
                }
                return None, cache

        class Model:
            vlm_with_expert = VlmWithExpert()

            def embed_prefix(self, images, image_masks, tokens, masks, state):
                testcase.assertTrue(masks.dtype is torch.bool)
                del images, image_masks, tokens, masks, state
                return (
                    torch.zeros((1, 3, 8)),
                    torch.tensor([[True, False, True]]),
                    torch.zeros((1, 3), dtype=torch.bool),
                )

        class SourcePolicy:
            model = Model()
            _queues = {}

            def _prepare_batch(self, batch):
                return batch

            def prepare_images(self, batch):
                del batch
                return [torch.zeros((1, 1, 2, 2))], [torch.ones((1,), dtype=torch.bool)]

            def prepare_state(self, batch):
                del batch
                return torch.zeros((1, 4))

            def reset(self):
                self._queues = {}

        batch = {
            "observation.language.tokens": torch.zeros((1, 2), dtype=torch.long),
            "observation.language.attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        provider = LeRobotSmolVLACacheProvider(SourcePolicy(), expert_layers=2)
        result = provider(batch)

        self.assertEqual(result.keys.shape, (2, 3, 320))
        self.assertEqual(result.values.shape, (2, 3, 320))
        np.testing.assert_array_equal(result.keys[1], 1.0)
        np.testing.assert_array_equal(result.values[0], 1.0)
        np.testing.assert_array_equal(result.mask, [1, 0, 1])
