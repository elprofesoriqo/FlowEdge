# ADR 0008: Lock-Free Thread Pool

## Status
Accepted

## Context
Standard threaded implementations (`std::function`, `std::mutex`) cause millisecond-scale context-switch jitter per layer. For low-latency edge inference, this is unacceptable.

## Decision
Implemented a custom `ThreadPool` using an SPMC lock-free ring buffer and `std::atomic`. 

- **Hybrid waiting:** Workers spin for a bounded window via `_mm_pause()` (x86) / `yield` (NEON),
  then park with C++ `atomic::wait`. A monotonically increasing work epoch prevents lost wakeups.
- **Affinity:** Threads are pinned to specific CPU cores at creation to maximize L1/L2 cache hits.
- **DRAM Saturation:** Capped at 8 workers to prevent memory bandwidth contention on laptop/edge CPUs.
- **Adaptive dispatch:** Automatic engines provision at most 4 workers. Matrix dimensions and weight
  representation select a power-of-two task tier per call; explicit C/Python overrides retain 0-8
  worker control for deployment tuning and regression measurements.

## Consequences
- **Positive:** Hot workers still avoid context switches between nearby kernels, while an idle engine
  stops consuming an entire core per worker.
- **Negative:** The first task after a long idle may pay an OS wakeup. The bounded spin window is the
  latency-versus-power tuning point.
