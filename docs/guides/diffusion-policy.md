# Diffusion Policy

FlowEdge supports the LeRobot `diffusion_pusht` action head as a fixed-shape CPU path.

| Supported | Boundary |
|---|---|
| ConditionalUnet1D, Conv1D/ConvTranspose1D, GroupNorm, Mish, FiLM | Batch size 1 |
| `squaredcos_cap_v2`, epsilon prediction | Fixed horizon and widths |
| Deterministic DDIM; seeded DDPM | External observation encoder |
| MIN_MAX action un-normalization | CPU only; no dynamic scheduler/horizon |

Reference shape: horizon 16, action dimension 2, action steps 8, observation steps 2, U-Net widths
512/1024/2048, timestep width 128, 100 training timesteps.

## Data boundary

```{mermaid}
flowchart LR
  Obs[Images + state] --> Enc[LeRobot preprocessing / encoder]
  Enc --> Cond[Flattened condition]
  Noise[Float32 noise] --> FE[FlowEdge Diffusion head]
  Cond --> FE
  FE --> Horizon[Full action horizon]
  Horizon --> Slice[LeRobot action slice]
  Slice --> Robot[Controller]
```

FlowEdge does not run the ResNet18 image encoder. For the pinned PushT policy, the external encoder
produces two time steps × 66 features = 132 condition values. The converter derives and validates
this width from checkpoint projections.

## Convert and sample

```bash
hf download lerobot/diffusion_pusht \
  --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht
python convert/convert.py models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
./build/diffusion_sample models/diffusion_pusht.flowedge.safetensors 10
```

Use `--processor` for a non-standard processor file. Only action un-normalization statistics are
imported; observation preprocessing remains in LeRobot.

LeRobot normally executes:

```text
actions[observation_steps - 1 : observation_steps - 1 + action_steps]
```

## Verify and benchmark

```bash
PYTHONPATH=build python scripts/verify_diffusion.py build
./build/flowedge_diffusion_latency_bench \
  models/diffusion_pusht.flowedge.safetensors 10 100
```

The verifier checks conversion, denoising, DDIM/DDPM determinism, metadata, and un-normalization
against an independent PyTorch reference. It reports explicit tolerances and observed error.

## Runtime boundary

| Fixed today | Not included |
|---|---|
| Float32 convolution/normalization; BF16 2D linear weights | Training, image encoding, arbitrary schedulers |
| Fixed shapes and one sample at a time | Dynamic horizons and accelerator backends |
| Caller-owned `sample_diffusion_into` output | ISA-specialized convolution (next optimization) |
