"""LeRobot processor factory for the FlowEdge deployment plugin."""

from lerobot.processor import (
    DeviceProcessorStep,
    IdentityProcessorStep,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)


def make_flowedge_pre_post_processors(config, dataset_stats=None):
    del dataset_stats
    return (
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device=config.device)],
            name="flowedge_preprocessor",
        ),
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu"), IdentityProcessorStep()],
            name="flowedge_postprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
