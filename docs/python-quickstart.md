# Python

CPU wheels for CPython 3.10–3.12 ship with GitHub Releases. They are not on PyPI yet. CUDA is a source build (`FLOWEDGE_BACKEND=cuda`), not a wheel extra.

## Install a wheel

1. Open [v0.1.1](https://github.com/elprofesoriqo/FlowEdge/releases/tag/v0.1.1).
2. Download the wheel for your OS and CPython from the linked [cibuildwheel run](https://github.com/elprofesoriqo/FlowEdge/actions/runs/35243534582) (manylinux x86_64 / aarch64, Windows AMD64, macOS ARM). Pinning those files onto the Release is [#184](https://github.com/elprofesoriqo/FlowEdge/issues/184).
3. Install it:

```bash
python -m pip install flowedge-0.1.1-*.whl
```

| Platform | Wheel tag |
|---|---|
| Linux x86_64 | `manylinux_2_28_x86_64` |
| Linux aarch64 | `manylinux_2_28_aarch64` |
| Windows | `win_amd64` |
| macOS ARM | `macosx_arm64` |

From a clone, `python -m pip install .` still compiles with a C++23 toolchain. That is the source path, not the first-run path.

## Sample flow matching

```bash
mkdir -p models
curl -L "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors" \
  -o models/mamba_flow.safetensors
python examples/core/flow_sample.py models/mamba_flow.safetensors euler 10
```

```python
import numpy as np, flowedge
e = flowedge.Engine("models/mamba_flow.safetensors")
a = e.sample(prefix=np.array([1, 2, 3, 4], dtype=np.int32),
             noise=np.random.randn(e.action_dim).astype("float32"),
             steps=10, method="euler")
```

That prefix is the ULP gate (~1e-6 rel vs PyTorch), not a policy p50.

Load may allocate once. Supported hot paths allocate nothing after init. One engine, one stream. The published wheel backend is CPU.

Zero-copy `run_into` / `sample_into` live in the [Python API](api/python).

## CUDA

Wheels do not include CUDA kernels. **v0.1.1 is a CPU wheel, not PyPI.** CUDA is a source build (`FLOWEDGE_BACKEND=cuda`). The first GPU user has to survive `nvcc` plus WSL or MSVC; a 4 GB card must not brick `fe_engine_load`.

On Windows, `nvcc` needs an MSVC-compatible host compiler. The MinGW/Clang tree used for CPU Release builds cannot host `nvcc`. Use WSL or MSVC.

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda -DFLOWEDGE_PYTHON=ON
cmake --build build-cuda -j
python -c "import flowedge; e=flowedge.Engine('models/diffusion_pusht.flowedge.safetensors'); print(flowedge.cuda, e.cuda_resident)"
```

`flowedge.cuda` is compile-time. `Engine.cuda_resident` is runtime: 1 only if Mamba / flow / Diffusion Policy weights actually uploaded. A GTX 1650 4 GB WDDM/WSL host often cannot `cudaMalloc` the PushT U-Net (~959 MiB). Load then keeps **CPU kernels**; that is not an engine failure. `FLOWEDGE_CUDA_REQUIRED=1` fails closed if attach did not happen. `FLOWEDGE_CUDA_FORCE_HOST=1` skips the doomed upload.

`--device cuda` in the LeRobot adapter requires both a CUDA Core binary and `cuda_resident`. The published CUDA headline stays GTX 1650 matched replay **131 vs 345 ms**. Fusion / BF16 stay ungated until isolate **and** native DDIM / policy p50 move on a card that can load the U-Net.

See [CUDA](architecture/cuda).

## PyPI (not published)

`pip install flowedge` is not live. On 2026-09-19 `https://pypi.org/pypi/flowedge/json` returned 404, so the name is still free. [publish.yml](https://github.com/elprofesoriqo/FlowEdge/blob/main/.github/workflows/publish.yml) already builds the same CPU wheels and can upload with Trusted Publishing when the repository variable `FLOWEDGE_PUBLISH_PYPI` is `true`. CUDA stays a source backend after that upload; do not advertise a CUDA extra that the wheel does not contain.

Maintainer checklist: PyPI Trusted Publisher for `flowedge` → GitHub environment `pypi` → set `FLOWEDGE_PUBLISH_PYPI=true` → publish a tagged release.
