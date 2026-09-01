# ADR 0003: Public API boundary

Status: Accepted. Scope: `src/core/api/`.

## Context

The engine ships as a library consumed over FFI from C, Rust, Go, or a ROS2 node. Consumers must not see internals. Nothing runtime-relevant may be baked in. The compute backend must stay swappable.

## Decision

- `src/core/api/` is a C-ABI over an opaque handle `fe_engine`. Internals live behind the handle.
- Everything runtime-relevant is a parameter. The model path and input come from the caller.
- The handle owns the whole runtime: slab, arena, tensor table, model, head. The slab is sized from the file.
- Exceptions never cross into C. `fe_engine_load` catches OOM and returns `nullptr`. Run and sample are non-throwing and rewind arena scratch per call.
- Consumers include only `api/engine.h`. Everything else is internal.

## Consequences

- The backend can change behind the ABI without touching consumers.
- The boundary is robust. Bad path or OOM returns `nullptr`. Bad tokens return a non-zero code. No crash.
