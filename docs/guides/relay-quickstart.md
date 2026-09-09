# Relay quickstart

Relay is the optional local service around Core: bounded IPC, deadline admission, worker pools,
state migration, action gating, replay, and metrics. Shared memory is local to one host.

## Build and run the complete demo

```bash
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release \
  -DFLOWEDGE_RELAY=ON -DFLOWEDGE_BENCH=ON
cmake --build build-relay --parallel
FLOWEDGE_BUILD_DIR=build-relay ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

```{mermaid}
flowchart LR
  C[Client] -->|request ring| D[flowedge-relayd]
  D --> A[EDF + freshness]
  A --> W[Preallocated workers]
  W --> Core[FlowEdge Core]
  Core --> R[result ring]
  W --> O[Trace + metrics]
```

The demo proves one inference, one typed `rejected_deadline`, graceful shutdown, trace replay, and
Prometheus/JSON/OTLP exports. `rejected_deadline` is a normal result, not a broken connection.

## Manual processes

Terminal 1:

```bash
./build-relay/src/relay/flowedge-relayd \
  --model models/mamba_flow.safetensors \
  --create --workers 2 --threads 0 --placement compact
```

Terminal 2:

```bash
./build-relay/src/relay/flowedge-relayctl request \
  --model models/mamba_flow.safetensors
./build-relay/src/relay/flowedge-relayctl shutdown
```

## Starting settings

| Setting | Start | Meaning |
|---|---:|---|
| `--workers` | `2` | Mutable model lanes / outer workers |
| `--threads` | `0` | Caller-thread-only Core execution |
| `--placement` | `compact` | Favor local shared-weight reads; compare `spread` |
| `--nfe-ns` | `0` | Admission disabled until calibrated |
| `--admission-reserve-ns` | measured | Transport/controller safety margin |
| `--metrics-interval-ms` | `0` | Export at shutdown; periodic I/O is opt-in |

Calibrate on the deployment host under sustained load:

```bash
./build-relay/flowedge_relay_bench models/mamba_flow.safetensors 5000 0
./build-relay/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0
```

Use the measured p99 NFE cost as an input, then add a safety reserve. It is not portable across CPUs,
power states, or thermal conditions.

## Outcomes

| Outcome | Action |
|---|---|
| `inference` | Pass through the application safety gate |
| `rejected_stale` / `cancelled` | Drop older work |
| `rejected_deadline` / `expired` | Reduce work, add capacity, or defer |
| `rejected_capacity` | Apply backpressure; retry only if fresh |
| `failed` | Inspect daemon error, trace, and metrics |

| Service | Use |
|---|---|
| `flowedge-relayd` | Condition vectors → flow action chunks |
| `flowedge-jobd` | Managed iterative, streaming, and speculative jobs |

See [cooperative jobs](cooperative-jobs) for the generic backend contract.
