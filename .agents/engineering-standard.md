# FlowEdge C++23 engineering standard

## Decision filter

1. Remove work that does not need to exist.
2. Reuse an existing FlowEdge contract.
3. Prefer a zero-overhead standard or platform facility.
4. Write the smallest measured implementation.

Modern syntax is not a performance argument. Use C++23 where it strengthens the contract:
`std::span` for borrowed buffers, `std::expected` for setup failures, concepts for adapters,
`std::jthread` or atomic wait/notify for owned threads, and `std::bit_cast`, `std::byteswap`,
`std::endian`, or `std::mdspan` for portable data views. Do not add abstraction without a measured
or correctness benefit.

## Runtime contract

| Property | Rule |
|---|---|
| Allocation | Core compute and documented Relay hot paths allocate nothing after construction. |
| Ownership | Dynamic containers are allowed for fixed setup storage; reserve or resize once. |
| Concurrency | No blocking mutex in runtime paths. Every atomic operation names its memory order. |
| Cache layout | Isolate independently contended control state at a 64-byte cache boundary. |
| Data access | Prefer typed values and spans; do not use aliasing or unaligned pointer tricks. |
| Errors | Setup returns `std::expected`; hot paths return compact enums or values. |
| Exceptions | Catch at setup boundaries. Runtime callbacks are `noexcept`. |

Acquire/release edges must publish an identifiable payload. Use relaxed ordering for counters and
thread-owned metadata. Never use default sequential consistency.

## Style

- C++23, leading return types, uniform initialization, snake-case functions and variables.
- Concepts instead of SFINAE; zero-copy views instead of owning parameters.
- Comments explain a constraint or tradeoff in one short sentence.
- Keep `FlowEdge::Core` independent. Relay may depend on Core; Core never depends on Relay.
- Preserve the current layout: `src/core/{api,arena,heads,kernels,loader,models,protocol,runtime}` and
  `src/relay/{adapters,client,jobs,protocol,scheduler,shared_memory,telemetry,worker}`.

## Performance evidence

Before optimizing, name the cost and capture a same-host Release baseline. Add or extend the smallest
benchmark that isolates it. Record median before/after results for Windows and Linux, verify zero
hot-path allocations, and keep exact reproduction commands in `docs/performance.md`.

Every new runnable surface belongs in `scripts/verify_all.sh`. Public Relay APIs must also compile and
run through `test/install_consumer`.
