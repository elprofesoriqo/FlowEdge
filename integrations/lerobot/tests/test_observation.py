from collections import deque
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("lerobot") and importlib.util.find_spec("torchvision"),
    "requires optional visual LeRobot runtime",
)
class ObservationTests(unittest.TestCase):
    def test_history_is_padded_copied_and_advanced_in_source_feature_order(self):
        import torch
        from flowedge_lerobot.observation import DiffusionObservationEncoder

        encoder = object.__new__(DiffusionObservationEncoder)
        encoder.config = SimpleNamespace(
            input_features={
                "observation.state": SimpleNamespace(shape=(2,)),
                "observation.image": SimpleNamespace(shape=(3, 2, 2)),
            },
            image_features={"observation.image": None},
            n_obs_steps=2,
            env_state_feature=None,
        )
        encoder.history = deque(maxlen=2)
        encoder.rgb_encoder = lambda images: images.mean(dim=(1, 2, 3)).unsqueeze(-1)
        batch = {
            "observation.state": torch.tensor([[1.0, 2.0]]),
            "observation.image": torch.ones(1, 3, 2, 2),
        }
        encoder.observe(batch, normalized=True)
        np.testing.assert_array_equal(encoder.condition(), [1, 2, 1, 1, 2, 1])
        batch["observation.state"].fill_(3)
        batch["observation.image"].fill_(4)
        encoder.observe(batch, normalized=True)
        np.testing.assert_array_equal(encoder.condition(), [1, 2, 1, 3, 3, 4])
        encoder.reset()
        with self.assertRaises(ValueError):
            encoder.condition()

    def test_normalization_requires_complete_valid_statistics(self):
        from lerobot.configs.types import FeatureType, NormalizationMode
        from flowedge_lerobot.observation import validate_stats

        config = SimpleNamespace(
            input_features={
                "observation.state": SimpleNamespace(type=FeatureType.STATE)
            },
            normalization_mapping={"STATE": NormalizationMode.MIN_MAX},
        )
        for stats in ({}, {"observation.state": {"min": [1.0], "max": [1.0]}}):
            with self.assertRaises(ValueError):
                validate_stats(config, stats)

    def test_processor_uses_supplied_statistics_without_unnormalizing_native_actions(
        self,
    ):
        import torch
        from unittest.mock import patch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from flowedge_lerobot.observation import observation_processor

        config = SimpleNamespace(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,))
            },
            output_features={},
            device="cpu",
            normalization_mapping={"STATE": NormalizationMode.MIN_MAX},
        )
        stats = {
            "observation.state": {"min": torch.zeros(2), "max": torch.full((2,), 10.0)}
        }
        with patch("flowedge_lerobot.observation.source_config", return_value=config):
            processor = observation_processor("unused", stats)
        normalized = processor({"observation.state": torch.tensor([0.0, 10.0])})
        torch.testing.assert_close(
            normalized["observation.state"], torch.tensor([[-1.0, 1.0]])
        )

    def test_source_pair_rejects_changed_configuration(self):
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        from safetensors.numpy import save_file
        from flowedge_lerobot.observation import validate_source_pair

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.safetensors"
            save_file({"fixture": np.zeros(1, dtype=np.float32)}, source)
            (root / "config.json").write_text("{}", encoding="utf-8")
            converted = root / "converted.safetensors"
            save_file(
                {"fixture": np.zeros(1, dtype=np.float32)},
                converted,
                metadata={
                    "flowedge.source_sha256": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest(),
                    "flowedge.source_config_sha256": hashlib.sha256(b"{}").hexdigest(),
                },
            )
            validate_source_pair(converted, root)
            (root / "config.json").write_text(
                json.dumps({"changed": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_source_pair(converted, root)
