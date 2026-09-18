# CUDA backend

`FLOWEDGE_BACKEND=cuda` links the ISA kernel surface (`matmul`, activations,
causal `conv1d`, `rmsnorm`, fused scan host reference) through
`src/core/kernels/cuda/`. CMake fails closed without `nvcc`.

This is not a device-resident engine. Each kernel still borrows host `std::span`
buffers, copies into grow-only device scratch, launches, and copies back.
Dense Diffusion Policy ops stay in `kernels/diffusion_ops.cc` on the CPU. Flow
and DDIM heads remain CPU until a later PR keeps weights and scratch on device
for the whole sample.

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda
cmake --build build-cuda -j
```

See [#160](https://github.com/elprofesoriqo/FlowEdge/issues/160). Device-resident
flow is [#161](https://github.com/elprofesoriqo/FlowEdge/issues/161); DP kernels
[#162](https://github.com/elprofesoriqo/FlowEdge/issues/162); matched GPU replay
[#163](https://github.com/elprofesoriqo/FlowEdge/issues/163); LeRobot CUDA
[#164](https://github.com/elprofesoriqo/FlowEdge/issues/164). Do not publish a
CUDA vs PyTorch CUDA policy number until that replay JSON exists.
