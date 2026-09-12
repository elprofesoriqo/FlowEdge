"""LeRobot-discoverable FlowEdge deployment plugin."""

try:
    import lerobot  # noqa: F401
except ImportError as error:
    raise ImportError("install lerobot to use lerobot_policy_flowedge") from error

from .configuration_flowedge import FlowEdgeConfig
from .modeling_flowedge import FlowEdgePolicy
from .processor_flowedge import make_flowedge_pre_post_processors

__all__ = ["FlowEdgeConfig", "FlowEdgePolicy", "make_flowedge_pre_post_processors"]
