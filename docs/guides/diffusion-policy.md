# Diffusion Policy

A diffusion policy treats the **action horizon** as the thing being denoised.
LeRobot trains a Conv1D U-Net that predicts noise given a noisy chunk, a
timestep, and an observation condition. FlowEdge runs that U-Net in a fixed
arena (DDIM by default, seeded DDPM for reference) and un-normalizes with the
checkpoint MIN_MAX stats. The RGB encoder stays in LeRobot.

```{image} ../_static/figures/policies.svg
:alt: Flow matching ODE versus Diffusion Policy DDIM
:class: fe-fig
```

Fixed-memory `diffusion_pusht` contract below. For the matched PyTorch p50 and
the 10 ms miss count, see [Performance](../performance.md).

```{image} ../_static/figures/convert.svg
:alt: convert a checkpoint once
:class: fe-fig
```

```{image} ../_static/figures/workflow.svg
:alt: convert, load arena, integrate, period, act
:class: fe-fig
```

Contract: fixed horizon / action dim / U-Net widths; Conv1D + ConvTranspose1D; GroupNorm, Mish, sinusoidal timestep, FiLM; `squaredcos_cap_v2` epsilon; DDIM / seeded DDPM; MIN_MAX un-normalize.

Reference: horizon 16, action dim 2, action steps 8, observation steps 2, widths 512/1024/2048, timestep 128, 100 train steps. Pinned revision `84a7c23178445c6bbf7e1a884ff497017910f653`. Newer LeRobot migrations may put action stats in `policy_postprocessor.json`; the converter reads `action.min` / `action.max` from `unnormalizer_processor`. MIN_MAX only.

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
python -m flowedge_dev pipeline convert models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
./build/diffusion_sample models/diffusion_pusht.flowedge.safetensors 10
```

For a non-standard processor filename, pass it explicitly with
`--processor path/to/policy_postprocessor.json`. Observation preprocessing
remains outside FlowEdge; only action un-normalization statistics are imported
from the processor state.

The output contains the whole 16-step denoised horizon in original PushT action
units. LeRobot normally executes only `n_action_steps`, beginning at index
`n_obs_steps - 1`; keep that slicing policy in the controller.

## Verify and benchmark

`tools/verification/verify_diffusion.py` (`python -m flowedge_dev verify diffusion`) creates a small LeRobot-shaped checkpoint, runs the
converter, loads the Python extension, and checks denoiser, DDIM, DDPM,
determinism, metadata, and un-normalization against an independent PyTorch
reference. Current explicit tolerances are `2e-4` for one denoiser pass, `4e-4`
for DDIM, and `8e-4` for seeded DDPM; observed errors are printed.

```bash
PYTHONPATH=build python -m flowedge_dev verify diffusion build
PYTHONPATH=build python -m flowedge_dev verify diffusion-public \
  models/diffusion_pusht/model.safetensors \
  models/diffusion_pusht.flowedge.safetensors --build build
```

The former C++ latency runner generated its own condition and noise, so it was
removed. Use the matched visual-policy replay in the [performance guide](../performance.md)
for end-to-end measurements with captured observations and source preprocessing.

## Limits

Batch size is one. Shapes are fixed at conversion. Convolution and normalization
weights remain F32; two-dimensional linear weights may be BF16. The scalar
diffusion kernels use fixed-shape loop specializations and the existing CPU
thread pool; ISA-specialized convolution is still planned. Training, image encoding, arbitrary schedulers,
dynamic horizons, and accelerator backends are outside this initial path.
