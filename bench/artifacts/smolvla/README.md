# SmolVLA recorded-cache replay

| File group | Evidence |
| --- | --- |
| `eslab-frame-000000.capture.*` | One recorded public LeRobot v3 frame, task tokens, and seeded flow noise. |
| `*.source-action.*` | Upstream `SmolVLAPolicy` 50 by 6 action chunk from that capture. |
| `*.expert-reference.*`, `*.trajectory-reference.*` | Upstream VLM K/V cache plus one expert velocity and ten-step Euler target. |
| `*.cached-expert.json`, `*.cached-trajectory.json` | FlowEdge replay reports for the cached expert boundary. |

The capture manifest pins the source dataset revision, license, parquet/video
digests, task, parquet and video frame indices, and noise seed. It maps the recorded top and wrist
cameras to two configured source keys without synthesizing, resizing, or
duplicating pixels.

This is cached-VLM action-expert parity evidence only. It does not establish
native VLM/preprocessing parity, end-to-end SmolVLA latency, control quality,
or task success.
