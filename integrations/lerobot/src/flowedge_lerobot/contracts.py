"""Backend-neutral deployment contracts used by integration adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ActionChunkAdapter(Protocol):
    """Minimal lifecycle and action-chunk surface for a deployment integration.

    Integrations may add policy-specific inputs, but the runtime boundary stays a resettable
    adapter that returns a bounded action chunk. No LeRobot or accelerator type is required here.
    """

    @property
    def action_dim(self) -> int: ...

    @property
    def action_steps(self) -> int: ...

    def reset(self) -> None: ...

    def predict_action_chunk(
        self, condition: np.ndarray, noise: np.ndarray, **kwargs: object
    ) -> np.ndarray: ...
