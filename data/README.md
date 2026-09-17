# Kernel mix datasets

This folder is for playing with the numbers behind the CPU Diffusion Policy note
in `docs/notes/substack-cpu-diffusion-policy.md`. Generated parquet/json/csv
files stay local (see `.gitignore`). The committed sample under `samples/` is
enough to run the explorer without a rebuild.

## Why run the benches

`flowedge_diffusion_breakdown` times the U-Net's real shapes, not a square GEMM.
Each row is a median, multiplied by how often that kernel runs in a 10-step DDIM
sample. That product is the only ranking we trust when choosing the next kernel
patch. A faster Mish that saves 8 ms does not matter while the 2048-channel
conv still estimates ~300 ms. On a laptop the same binary can move 100 ms
between a hot and a cool run; `data/explore.py a.parquet --vs b.json` is how
you tell a kernel delta from a clock sag. Do not overwrite the matched
PyTorch replay artifact from a slower mix run.

The optional native lines need the converted PushT checkpoint. The mix lines do
not.

## Export

From the repo root, after a Release build that includes `FLOWEDGE_BENCH=ON`:

```powershell
python tools/benchmark/export_kernel_mix.py --build-dir build-win-clang
python tools/benchmark/export_kernel_mix.py --build-dir build-win-clang `
  --checkpoint models/diffusion_pusht.flowedge.safetensors
```

That writes `data/kernel_mix.json` and, if pyarrow is installed,
`data/kernel_mix.parquet`. CSV is always written as a fallback.
`pip install pyarrow` if you want parquet. `--from-json` converts an existing
JSON dump without re-running the binary.

## Explore

```powershell
python data/explore.py data/samples/kernel_mix.example.json
python data/explore.py data/kernel_mix.parquet
python data/explore.py data/kernel_mix.parquet --vs data/samples/kernel_mix.example.json
```

`explore.py` prints the mix sorted by estimated milliseconds per sample, and the
arithmetic intensity of the short-M GEMMs:

\[
I \approx \frac{2 M N K}{4(MK + NK + MN)} \approx \frac{M}{2}
\]

when weights dominate. For \(M=4\) that is 2 FLOP/byte, which is why inner-loop
tweaks stop moving the median.
