import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("lerobot") and importlib.util.find_spec("torchvision"),
    "requires the full optional LeRobot runtime",
)
class PluginTests(unittest.TestCase):
    def test_plugin_registers_flowedge_policy(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        self.assertIsNotNone(PreTrainedConfig.get_choice_class("flowedge"))


if __name__ == "__main__":
    unittest.main()
