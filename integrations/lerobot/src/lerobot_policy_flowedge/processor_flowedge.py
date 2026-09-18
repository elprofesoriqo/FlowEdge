"""LeRobot processor factory for the FlowEdge deployment plugin."""

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from flowedge_lerobot.device import config_device


def make_flowedge_pre_post_processors(config, dataset_stats=None):
    config.validate_features()
    if config.input_mode == "visual":
        from flowedge_lerobot.observation import observation_processor

        preprocessor = observation_processor(
            config.source_checkpoint_path, dataset_stats
        )
    else:
        if dataset_stats:
            raise ValueError(
                "encoded conditions must already be normalized; dataset_stats requires visual input"
            )
        preprocessor = PolicyProcessorPipeline(
            steps=[
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=config_device(config)),
            ],
            name="flowedge_preprocessor",
        )
    return (
        preprocessor,
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu"), IdentityProcessorStep()],
            name="flowedge_postprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
