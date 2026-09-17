# Kernel mix datasets

![do not mix the evidence](../docs/_static/figures/benches.svg)

Numbers behind the CPU Diffusion Policy mix. Generated parquet/json/csv stay gitignored; `samples/` is enough to explore.

`flowedge_diffusion_breakdown` times real U-Net shapes. Rank kernels by median × calls per 10-step DDIM, not by a square GEMM. `data/explore.py a.parquet --vs b.json` separates a kernel delta from clock sag.

```bash
python -m flowedge_dev bench mix --build-dir build-win-clang
python -m flowedge_dev bench mix --build-dir build-win-clang \
  --checkpoint models/diffusion_pusht.flowedge.safetensors
python data/explore.py data/samples/kernel_mix.example.json
```

Writes `data/kernel_mix.json` (and parquet if pyarrow is installed). Mix lines do not need the PushT checkpoint; native DDIM lines do.
