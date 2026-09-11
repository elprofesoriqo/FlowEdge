# Examples

Build with CMake, then choose the smallest example for the contract you need.

| Area | Start with | Continue with |
|---|---|---|
| C++ policy inference | `core/flow_sample.cc` | `core/diffusion_sample.cc` |
| Python policy inference | `core/flow_sample.py` | `integrations/lerobot/` |
| External observations | `core/external_flow_sample.cc` | `core/streaming_snapshot.cc` |
| Mamba backbone | `core/mamba_forward.cc` | `relay/mamba_relay_stream.cc` |
| Relay client | `relay/relay_client_sample.cc` | `relay/action_delivery_sample.cc` |
| Generic jobs | `relay/cooperative_job_sample.cc` | `relay/routed_job_sample.cc` |

`core/` uses `FlowEdge::Core`. `relay/` requires `-DFLOWEDGE_RELAY=ON`. Executables keep
their source filename without `.cc`; for example, `core/flow_sample.cc` builds `flow_sample`.
