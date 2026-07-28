# Roadmap

## Done

- Mamba backbone with streaming decode.
- Flow-matching head. Euler, Heun, RK4.
- CPU kernels. AVX2, NEON, scalar.
- C-ABI, Python module, find_package packaging.
- Checkpoint converter.
- ULP gate against PyTorch in CI.

## Next

- Heads: Diffusion Policy, ACT, VQ-BeT, pi0.
- Backbone: Transformer. Unlocks the transformer heads.
- Weight traffic: BF16 and INT8 weights, then threading. This is the performance work that matters. See [ADR 0007](decisions/0007-roofline).
- Backends: CUDA, Tenstorrent.
- Observation encoders, which every real vision policy needs before it runs end to end. See [ADR 0006](decisions/0006-obs-encoder-out-of-scope).