# Maintainer tools

| Need | Location |
|---|---|
| Inspect a checkpoint | `checkpoint_inspect.cc` (`flowedge-inspect`) |
| Compare or publish benchmarks | `benchmark/` |
| Run model and converter parity checks | `verification/` |

Most users only need the stable commands in `scripts/`. The complete maintainer gate is
`scripts/verify_all.sh`.
