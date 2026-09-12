"""Inference-only LeRobot policy wrapper around a converted FlowEdge checkpoint."""

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from flowedge_lerobot import FlowEdgeDiffusionPolicy

from .configuration_flowedge import FlowEdgeConfig


class FlowEdgePolicy(PreTrainedPolicy):
    config_class = FlowEdgeConfig
    name = "flowedge"

    def __init__(self, config: FlowEdgeConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        config.validate_features()
        self._policy = FlowEdgeDiffusionPolicy.from_checkpoint(config.checkpoint_path)
        self._noise = np.zeros(
            (self._policy.metadata.horizon, self._policy.action_dim), dtype=np.float32
        )
        self._chunk = np.empty(
            (self._policy.action_steps, self._policy.action_dim), dtype=np.float32
        )

    def get_optim_params(self) -> dict:
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def reset(self) -> None:
        self._policy.reset()

    def forward(self, batch):
        del batch
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def predict_action_chunk(self, batch, **kwargs):
        condition = self._condition(batch)
        self._policy.predict_action_chunk_into(
            condition, self._noise, self._chunk, **kwargs
        )
        return torch.from_numpy(self._chunk.copy()).unsqueeze(0)

    def select_action(self, batch, **kwargs):
        return self.predict_action_chunk(batch, **kwargs)[:, 0]

    def _condition(self, batch) -> np.ndarray:
        state = batch.get(OBS_STATE)
        if state is None or state.shape[0] != 1:
            raise ValueError(
                "FlowEdge plugin requires a batch of one observation.state"
            )
        return state.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
