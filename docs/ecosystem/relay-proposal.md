# FlowEdge Relay

Relay is the optional local systems layer around `FlowEdge::Core`.

## Why it exists

Robotics control loops need the newest valid action before a physical deadline. Relay supplies the
bounded local transport and lifecycle that a static inference library intentionally does not.

| Core owns | Relay owns |
|---|---|
| Model loading, kernels, solver, mutable inference state | Shared-memory rings and process boundary |
| Static arenas and zero-allocation hot path | Freshness, EDF/QoS admission, cancellation |
| C/C++/Python inference API | Worker pools, migration, action safety gate |
| Checkpoint/model identity | Replay traces and fixed-memory metrics |

## Data flow

```{mermaid}
flowchart LR
  Sensor[Sensor / VLA] -->|condition + timestamp| Client[Relay client]
  Client -->|SPSC shared memory| Daemon[Relay daemon]
  Daemon --> Admit[Freshness + EDF/QoS]
  Admit --> Workers[Preallocated Core workers]
  Workers --> Candidate[Action chunk]
  Candidate --> Gate[Bounds + overlap + freshness]
  Gate --> Robot[Robot adapter]
  Daemon --> Evidence[Trace + metrics]
```

## Current surface

| Area | Contract |
|---|---|
| Messages | Versioned condition/action/job records with model identity and generation |
| Transport | Checksummed, fixed-capacity SPSC shared-memory rings |
| Scheduling | Bounded EDF; optional calibrated NFE admission; class reservations |
| Workers | 1–8 mutable lanes, immutable shared weights, compact/spread placement |
| State | Canonical model/schema/session-bound capsules; exact Mamba continuation |
| Operations | `flowedge-relayd`, `flowedge-relayctl`, `flowedge-jobd`, `flowedge-jobctl` |
| Evidence | Portable trace inspection/replay; Prometheus, JSON, OTLP/HTTP JSON |
| Recovery | Drain, migration, quarantine, explicit recovery, bounded event buffer |

## Build and measure

```bash
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release \
  -DFLOWEDGE_RELAY=ON -DFLOWEDGE_TESTS=ON -DFLOWEDGE_BENCH=ON
cmake --build build-relay --parallel
ctest --test-dir build-relay --output-on-failure
FLOWEDGE_BUILD_DIR=build-relay ./scripts/relay_demo.sh models/mamba_flow.safetensors
./build-relay/flowedge_relay_bench models/mamba_flow.safetensors 5000 0
```

Trace tools:

```bash
./build-relay/src/relay/flowedge-relay-trace inspect run.trace --jsonl
./build-relay/src/relay/flowedge-relay-trace replay run.trace \
  --model models/mamba_flow.safetensors --tolerance 1e-5
```

## Deployment checklist

| Step | Action |
|---:|---|
| 1 | Start with `--threads 0`, `--workers 2`, and `--placement compact` |
| 2 | Measure p99 NFE cost on the target CPU and thermal policy |
| 3 | Add `--nfe-ns` plus a measured admission reserve |
| 4 | Re-test stale, deadline, capacity, expiry, drain, and recovery outcomes |
| 5 | Keep final robot safety and emergency-stop logic outside Relay |

`--nfe-ns 0` disables predictive deadline rejection; expiry-at-dispatch remains active. One
`RelayClient` endpoint is one SPSC producer/consumer pair. Use an application-side MPSC gate when
multiple producers are required.

## Beyond robotics

The same bounded worker contract fits latent decoders, speculative SSM/LLM branches, iterative
generative heads, and other stateful model stages. Domain-specific adapters stay outside Core.
