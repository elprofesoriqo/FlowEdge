# ADR 0008: Lock-Free Thread Pool

## Status
Accepted

## Context
Standard threaded implementations (`std::function`, `std::mutex`) cause millisecond-scale context-switch jitter per layer. For low-latency edge inference, this is unacceptable.

## Decision
Implemented a custom `ThreadPool` using an SPMC lock-free ring buffer and `std::atomic`. 

- **Spinning vs Sleeping:** Workers actively spin via `_mm_pause()` (x86) / `yield` (NEON).
- **Affinity:** Threads are pinned to specific CPU cores at creation to maximize L1/L2 cache hits.
- **DRAM Saturation:** Capped at 8 workers to prevent memory bandwidth contention on laptop/edge CPUs.

## Consequences
- **Positive:** Zero context switch penalties; matrix multiplication scales linearly up to DRAM limits.
- **Negative:** Pinned threads consume 100% CPU during the forward pass. Acceptable for short, real-time bursts in robotics.
