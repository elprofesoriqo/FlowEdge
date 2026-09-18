# CUDA backend

`FLOWEDGE_BACKEND=cuda` links the ISA kernel surface (`matmul`, activations
including `mish`, causal `conv1d`, `rmsnorm`, fused scan host reference) through
`src/core/kernels/cuda/`. CMake fails closed without `nvcc`. On Windows, `nvcc`
needs an MSVC-compatible host compiler; the MinGW/Clang gnu-target tree used
for CPU Release builds cannot host `nvcc`. WSL or MSVC is the CUDA build.

The **flow head** and **Diffusion Policy head** are device-resident. `FlowHead`
uploads `flow.*` weights and ODE scratch once at load. `sample` /
`sampler_advance` copy condition and noise in, run Euler/Heun/RK4 on device, and
copy the action out. `DiffusionHead` uploads `dp.*` weights and U-Net/DDIM
scratch once. `denoise` / `sample` copy condition and the horizon in, run the
Conv1D U-Net and DDIM (seeded DDPM) on device, and copy the action or epsilon
out. No `cudaMalloc` after load. `kernels.h` stays host `std::span`; Mamba still
round-trips. Dense DP ops also remain available as host-span launches from the
same CUDA TU.

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda
cmake --build build-cuda -j
./build-cuda/flow_sample models/mamba_flow.safetensors euler 10
./build-cuda/diffusion_sample models/diffusion_pusht.flowedge.safetensors 10
```

Same checkpoint, same noise: compare those binaries to CPU `flow_sample` /
`diffusion_sample`. Matched full-policy CUDA vs PyTorch CUDA on GTX 1650 is
{download}`diffusion-pusht-cuda-replay.md <../../bench/artifacts/policy/diffusion-pusht-cuda-replay.md>`:
policy p50 420 ms vs 346 ms, max abs 4.58e-5, same CPU observation file.
The LeRobot CUDA period log is
{download}`diffusion-pusht-cuda-period10-hold.md <../../bench/artifacts/policy/diffusion-pusht-cuda-period10-hold.md>`:
10 ms hold missed 20 / 20 (step p50 589 ms). Not TensorRT/ONNX. Not Jetson/ARM.

WSL2, nvcc 12.0, GTX 1650 (`sm_75`): `FlowHead.*` and `DiffusionHead.*` including
`CudaResidentDoesNotMallocAfterLoad` pass. Same `mamba_flow` checkpoint and
noise, `flow_sample` Euler/Heun/RK4 `action[0..2]` matches a CPU binary at
printed precision. Same `diffusion_pusht` checkpoint and noise, 10-step DDIM
`diffusion_sample` matches the CPU horizon at printed precision (iostream
rounding on two components: `226.249` vs `226.25`). That is not a policy p50.

See [#160](https://github.com/elprofesoriqo/FlowEdge/issues/160). Device-resident
flow is [#161](https://github.com/elprofesoriqo/FlowEdge/issues/161); device-resident
DDIM is [#162](https://github.com/elprofesoriqo/FlowEdge/issues/162); matched GPU
replay [#163](https://github.com/elprofesoriqo/FlowEdge/issues/163); LeRobot CUDA
rollout is `--device cuda` on `flowedge-lerobot-rollout` with the same
`--on-miss` contract ([#164](https://github.com/elprofesoriqo/FlowEdge/issues/164)).
The RGB encoder stays in LeRobot. Not Jetson/ARM.
