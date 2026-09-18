# CUDA backend

`FLOWEDGE_BACKEND=cuda` links the ISA kernel surface (`matmul`, activations
including `mish`, causal `conv1d`, `rmsnorm`, fused scan host reference) through
`src/core/kernels/cuda/`. CMake fails closed without `nvcc`. On Windows, `nvcc`
needs an MSVC-compatible host compiler; the MinGW/Clang gnu-target tree used
for CPU Release builds cannot host `nvcc`. WSL or MSVC is the CUDA build.

The **flow head** is device-resident: `FlowHead` uploads `flow.*` weights and
ODE scratch once at load. `sample` / `sampler_advance` copy condition and noise
in, run Euler/Heun/RK4 on device, and copy the action out. No `cudaMalloc` after
load. `kernels.h` stays host `std::span`; Mamba still round-trips. Dense Diffusion
Policy ops live in the CUDA TU and also copy host spans; a device-resident DDIM
head is the rest of [#162](https://github.com/elprofesoriqo/FlowEdge/issues/162).

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda
cmake --build build-cuda -j
./build-cuda/flow_sample models/mamba_flow.safetensors euler 10
```

Same checkpoint, same noise: compare that binary to a CPU `flow_sample`. Do not
publish a CUDA vs PyTorch policy p50 (that is [#163](https://github.com/elprofesoriqo/FlowEdge/issues/163)).

WSL2, nvcc 12.0, GTX 1650 (`sm_75`): `FlowHead.*` including
`CudaResidentDoesNotMallocAfterLoad` pass. Same `mamba_flow` checkpoint and
noise, `flow_sample` Euler/Heun/RK4 `action[0..2]` matches a CPU binary at
printed precision. That is not a policy p50.

See [#160](https://github.com/elprofesoriqo/FlowEdge/issues/160). Device-resident
flow is [#161](https://github.com/elprofesoriqo/FlowEdge/issues/161); DP kernels
[#162](https://github.com/elprofesoriqo/FlowEdge/issues/162); matched GPU replay
[#163](https://github.com/elprofesoriqo/FlowEdge/issues/163); LeRobot CUDA
[#164](https://github.com/elprofesoriqo/FlowEdge/issues/164).
