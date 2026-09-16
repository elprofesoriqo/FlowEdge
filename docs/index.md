# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge runs a trained LeRobot policy in a fixed-memory C++ runtime. Convert the checkpoint, keep the encoder in LeRobot, and measure the control period. Current CPU Diffusion Policy is correct and not yet faster than PyTorch; that gap is the first engineering target.</p>

## Convert, run, measure

| Step | Entry point |
|---|---|
| Convert a pinned LeRobot checkpoint | [Converter](guides/converter) |
| Run the LeRobot plugin or rollout | [LeRobot adapter](guides/lerobot) |
| Measure p50, missed deadlines, RSS | [Performance](performance) |

`--policy.type=flowedge` is Diffusion Policy. `--policy.type=flowedge_smolvla` is the native action expert with a LeRobot VLM cache. Mamba + flow remains the CI fixture. Relay, cooperative jobs, and generic daemons are optional systems layers.

It is not a training framework and not a graph runtime. Every weight and every scratch buffer comes from a single arena, sized once at load, so the runtime path allocates nothing.

## Other surfaces

| Goal | Entry point |
|---|---|
| Keep an existing encoder and use only the action head | [Capabilities](capabilities) |
| Getting-started command list | [Getting Started](getting-started) |
| Deadline-aware inference between processes | [Relay quickstart](guides/relay-quickstart) |
| Choose a model artifact path | [Model import](guides/model-import) |
| Preflight a checkpoint before deployment | [Checkpoint preflight](guides/checkpoint-preflight) |

## Repository map

| Area | Contents |
|---|---|
| `src/core/` | Allocation-free model and solver runtime |
| `src/relay/` | Optional local scheduling and transport |
| `python/` and `integrations/lerobot/` | Python API and LeRobot deployment adapter |
| `convert/` and `tools/` | Checkpoint conversion and inspection |
| `examples/` | Small C++ and Python entry-point samples |
| `scripts/` | Build, test, verification, and maintainer benchmarks |
| `bench/` | Categorized kernel/runtime sources and evidence artifacts; see [benchmark map](benchmarks) |
| `cmake/` | Installed-package export template; generated `CMakeFiles/` is ignored |

## Implementation status

<div class="fe-grid">
  <div class="fe-card">
    <h4>User policies</h4>
    <ul><li class="done">Diffusion Policy (correct, slower than PyTorch)</li><li class="done">SmolVLA cached expert</li></ul>
  </div>
  <div class="fe-card">
    <h4>Runtime</h4>
    <ul><li class="done">CPU AVX2 / NEON</li><li class="done">FP32 / BF16</li><li class="done">Zero-alloc hot path</li></ul>
  </div>
  <div class="fe-card">
    <h4>LeRobot plugin</h4>
    <ul><li class="done"><code>flowedge</code></li><li class="done"><code>flowedge_smolvla</code></li><li class="done">hold / drop / raise</li></ul>
  </div>
</div>

## How it fits

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph TD
  W[".safetensors"] --> L[Loader]
  L --> A[Arena]
  T[tokens] --> B[Backbone]
  A --> B
  B --> C[conditioning vector]
  C --> H[Head]
  H --> ACT[action chunk]
  K["Kernels: CPU"] -.-> B
  K -.-> H
```

```{toctree}
:hidden:
:caption: Start
Overview <self>
getting-started
guides/lerobot
guides/converter
guides/policy-evaluation
performance
benchmarks
capabilities
```

```{toctree}
:hidden:
:caption: Architecture
architecture/overview
architecture/memory
architecture/loader
architecture/deployment-profile
architecture/kernels
architecture/backbones
architecture/heads
architecture/cooperative-execution
architecture/model-porting
decisions/index
```

```{toctree}
:hidden:
:caption: API
api/c-abi
api/python
```

```{toctree}
:hidden:
:caption: Guides
guides/diffusion-policy
guides/transformer-backbone
guides/checkpoint-preflight
guides/model-import
guides/edge-benchmarks
guides/deadline-profile
guides/sanitizers
guides/verification
guides/observability
guides/add-a-head
guides/add-a-backbone
```

```{toctree}
:hidden:
:caption: Systems (optional)
guides/relay-quickstart
guides/action-delivery
guides/cooperative-jobs
guides/generic-job-daemon
guides/deadline-flow
tenstorrent-program
```

```{toctree}
:hidden:
:caption: Reference
roadmap
product-direction
FlowEdge Relay <ecosystem/relay-proposal>
LeRobot edge-inference RFC draft <ecosystem/lerobot-rfc>
```
