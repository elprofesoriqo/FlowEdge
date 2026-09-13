"""LeRobot configuration for a converted FlowEdge Diffusion Policy."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("flowedge")
@dataclass
class FlowEdgeConfig(PreTrainedConfig):
    checkpoint_path: str = ""
    action_steps: int = 1

    def get_optimizer_preset(self) -> AdamWConfig:
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if self.robot_state_feature is None or self.action_feature is None:
            raise ValueError(
                "FlowEdge plugin requires observation.state input and action output features"
            )
        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path must name a converted FlowEdge checkpoint"
            )

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.action_steps))

    @property
    def reward_delta_indices(self):
        return None
