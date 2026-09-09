# Diffusion Policy

FlowEdge supports the LeRobot `diffusion_pusht` action head as a fixed-memory
`ConditionalUnet1D`. The first supported contract is intentionally narrow:

- fixed horizon, action dimension, and U-Net stage widths;
- Conv1D downsampling and ConvTranspose1D upsampling;
- GroupNorm, Mish, sinusoidal timestep embedding, and FiLM scale/bias;
- `squaredcos_cap_v2` with epsilon prediction;
- deterministic DDIM and seeded DDPM;
- MIN_MAX action un-normalization.

The reference checkpoint configuration is horizon 16, action dimension 2,
action steps 8, observation steps 2, U-Net widths 512/1024/2048, timestep width
128, and 100 training timesteps.

The documented conversion command is pinned to the legacy public checkpoint
revision `84a7c23178445c6bbf7e1a884ff497017910f653`. Newer LeRobot migrations
may move action normalization statistics into processor configuration; those
layouts are not accepted by this first fixed schema.

## Observation boundary

FlowEdge does not run the reference ResNet18 image encoder. Run LeRobot's
observation preprocessing and encoder externally, then pass the flattened
condition immediately before `DiffusionConditionalUnet1d`.

For `diffusion_pusht`, each of the two observation steps contributes the
normalized 2-value robot state followed by the 64-value spatial-softmax image
feature. Flatten the `[2, 66]` tensor in time order to obtain the 132-value
condition. The converter derives and validates this width from every FiLM
projection rather than hard-coding it.

## Convert and sample

```bash
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht
python convert/convert.py models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
./build/diffusion_sample models/diffusion_pusht.flowedge.safetensors 10
```

The output contains the whole 16-step denoised horizon in original PushT action
units. LeRobot normally executes only `n_action_steps`, beginning at index
`n_obs_steps - 1`; keep that slicing policy in the controller.

## Verify and benchmark

`scripts/verify_diffusion.py` creates a small LeRobot-shaped checkpoint, runs the
converter, loads the Python extension, and checks denoiser, DDIM, DDPM,
determinism, metadata, and un-normalization against an independent PyTorch
reference. Current explicit tolerances are `2e-4` for one denoiser pass, `4e-4`
for DDIM, and `8e-4` for seeded DDPM; observed errors are printed.

```bash
PYTHONPATH=build python scripts/verify_diffusion.py build
PYTHONPATH=build python scripts/verify_diffusion_public.py \
  models/diffusion_pusht/model.safetensors \
  models/diffusion_pusht.flowedge.safetensors --build build
FLOWEDGE_BENCH_CPU="Ryzen 9 7950X" \
  ./build/flowedge_diffusion_latency_bench \
  models/diffusion_pusht.flowedge.safetensors 10 100
```

The benchmark reports compiler, build type, threads, steps, and denoiser/full
DDIM p50/p95/p99 latency. Set `FLOWEDGE_BENCH_CPU` so saved results identify the
processor.

## Limits

Batch size is one. Shapes are fixed at conversion. Convolution and normalization
weights remain F32; two-dimensional linear weights may be BF16. The scalar
diffusion kernels use fixed-shape loop specializations and the existing CPU
thread pool; ISA-specialized convolution is still planned. Training, image encoding, arbitrary schedulers,
dynamic horizons, and accelerator backends are outside this initial path.
