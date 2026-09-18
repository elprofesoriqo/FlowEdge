# CUDA backend

`FLOWEDGE_BACKEND=cuda` links the ISA kernel surface (`matmul`, activations
including `mish`, causal `conv1d`, `rmsnorm`, device SSM scan) through
`src/core/kernels/cuda/`. CMake fails closed without `nvcc`. On Windows, `nvcc`
needs an MSVC-compatible host compiler; the MinGW/Clang gnu-target tree used
for CPU Release builds cannot host `nvcc`. WSL or MSVC is the CUDA build.

The **Mamba backbone**, **flow head**, and **Diffusion Policy head** are
device-resident. `Mamba` uploads `backbone.*` weights and layer scratch once at
load. `forward` / `decode` copy the token activations (and host decode state)
in, run the mixer on device, and copy the hidden state out. `FlowHead`
uploads `flow.*` weights and ODE scratch once at load. `sample` /
`sampler_advance` copy condition and noise in, run Euler/Heun/RK4 on device, and
copy the action out. `DiffusionHead` uploads `dp.*` weights and U-Net/DDIM
scratch once. `denoise` / `sample` copy condition and the horizon in, run the
Conv1D U-Net and DDIM (seeded DDPM) on device, and copy the action or epsilon
out. No `cudaMalloc` after load. `kernels.h` stays host `std::span` for the
generic op surface. Dense DP ops also remain available as host-span launches
from the same CUDA TU. Short PushT horizons launch with `block.x` matching `L`
so time-axis stores stay coalesced; `conv_transpose1d` recovers `it` from `ot`,
`k`, and stride instead of scanning `input_length`. L=4 dense conv splits `IC`
across a warp so the inner reduction is not serial. `flowedge_cuda_dp_mix`
isolates those shapes.
See [#174](https://github.com/elprofesoriqo/FlowEdge/issues/174) and
[#176](https://github.com/elprofesoriqo/FlowEdge/issues/176).

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda
cmake --build build-cuda -j
./build-cuda/flow_sample models/mamba_flow.safetensors euler 10
./build-cuda/mamba_forward models/mamba_flow.safetensors 1 2 3 4
./build-cuda/diffusion_sample models/diffusion_pusht.flowedge.safetensors 10
./build-cuda/flowedge_cuda_dp_mix models/diffusion_pusht.flowedge.safetensors
```

Same checkpoint, same noise: compare those binaries to CPU `flow_sample` /
`diffusion_sample`. Matched full-policy CUDA vs PyTorch CUDA on GTX 1650 is
{download}`diffusion-pusht-cuda-replay.md <../../bench/artifacts/policy/diffusion-pusht-cuda-replay.md>`:
policy p50 153 ms vs 361 ms, max abs 7.63e-5, same CPU observation file.
The LeRobot CUDA period log is
{download}`diffusion-pusht-cuda-period10-hold.md <../../bench/artifacts/policy/diffusion-pusht-cuda-period10-hold.md>`:
10 ms hold missed 20 / 20 (step p50 589 ms). Not TensorRT/ONNX. Not Jetson/ARM.

WSL2, nvcc 12.0, GTX 1650 (`sm_75`): `FlowHead.*`, `DiffusionHead.*`, and
`Mamba.*` including `CudaResidentDoesNotMallocAfterLoad` pass. Same
`mamba_flow` checkpoint, `mamba_forward` tokens `1 2 3 4` prints `out[0]=-1.10703`
on CUDA and CPU; `flow_sample` Euler/Heun/RK4 `action[0..2]` matches a CPU
binary at printed precision. Same `diffusion_pusht` checkpoint and noise,
10-step DDIM `diffusion_sample` matches the CPU horizon at printed precision
(iostream rounding on two components: `226.249` vs `226.25`).
`flowedge_cuda_dp_mix` native 10-step DDIM p50 on this card is 152 ms after L=4
split-K (321 ms after #174, 448 ms before). That is not a policy p50.

See [#160](https://github.com/elprofesoriqo/FlowEdge/issues/160). Device-resident
flow is [#161](https://github.com/elprofesoriqo/FlowEdge/issues/161); device-resident
DDIM is [#162](https://github.com/elprofesoriqo/FlowEdge/issues/162); matched GPU
replay [#163](https://github.com/elprofesoriqo/FlowEdge/issues/163); LeRobot CUDA
rollout is `--device cuda` on `flowedge-lerobot-rollout` with the same
`--on-miss` contract ([#164](https://github.com/elprofesoriqo/FlowEdge/issues/164));
device-resident Mamba is [#172](https://github.com/elprofesoriqo/FlowEdge/issues/172);
CUDA DP conv occupancy is [#174](https://github.com/elprofesoriqo/FlowEdge/issues/174);
L=4 split-K conv is [#176](https://github.com/elprofesoriqo/FlowEdge/issues/176).
The RGB encoder stays in LeRobot. Not Jetson/ARM.
