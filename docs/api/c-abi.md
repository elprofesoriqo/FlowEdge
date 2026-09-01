# C-ABI

The only public surface. C linkage, opaque handle.

```c
fe_engine* fe_engine_load(const char* path);
fe_engine* fe_engine_load_with_threads(const char* path, unsigned worker_threads);
void       fe_engine_free(fe_engine* e);

void   fe_engine_dims(const fe_engine*, size_t* d_model, size_t* n_layers);
size_t fe_engine_action_dim(const fe_engine*);
size_t fe_engine_condition_dim(const fe_engine*);
unsigned fe_engine_thread_count(const fe_engine*);

int fe_engine_run(fe_engine*, const int32_t* tokens, size_t n, float* out);
int fe_engine_sample(fe_engine*, const int32_t* tokens, size_t n,
                     const float* noise, size_t steps, int method, float* action);
int fe_engine_sample_condition(fe_engine*, const float* condition,
                               const float* noise, size_t steps,
                               int method, float* action);

int fe_engine_flow_begin(fe_engine*, const float* condition,
                         const float* noise, size_t steps, int method);
int fe_engine_flow_advance(fe_engine*, size_t step_budget, float* action,
                           size_t* steps_remaining);

int  fe_engine_step(fe_engine*, int32_t token, float* out);
void fe_engine_reset(fe_engine*);

size_t fe_engine_decode_state_bytes(const fe_engine*);
int fe_engine_export_decode_state(const fe_engine*, void* destination, size_t bytes);
int fe_engine_import_decode_state(fe_engine*, const void* source, size_t bytes);

const char* fe_engine_last_error(void);
```

Rules:

- Everything runtime-relevant is a parameter. Nothing is baked in.
- The handle owns the whole runtime. Free it with `fe_engine_free`.
- `fe_engine_load` reads `FLOWEDGE_THREADS=0..8` when present and otherwise uses a
  bandwidth-aware automatic default. `fe_engine_load_with_threads` bypasses the environment;
  zero selects caller-thread-only execution.
- No exception crosses the boundary. Errors return `nullptr` or a non-zero code.
- `method` is 0 for Euler, 1 for Heun, 2 for RK4.
- Prefer the named constants `FE_SOLVER_EULER`, `FE_SOLVER_HEUN`, and `FE_SOLVER_RK4` from
  `engine.h` instead of literal method values. Unknown values fail with a non-zero return code.
- `fe_engine_sample_condition` accepts the output of an encoder owned by another runtime. A
  checkpoint may therefore contain only `flow.*` tensors and no built-in backbone.
- `fe_engine_flow_begin` projects the condition once. Each `fe_engine_flow_advance` executes at
  most `step_budget` complete solver steps and reports how many remain. Starting a new solve
  replaces the previous one.
- Decode snapshots are raw, allocation-free state copies. Restore them only into an engine loaded
  from the identical checkpoint; they do not contain a schema or model digest.
- One handle has mutable scratch and sampler state and is not safe for concurrent calls. Use one
  engine per concurrently executing session.
- Prefill and token-conditioned sampling accept 1 to 512 tokens per call; streaming `step` has no
  growing sequence buffer.

## Link with CMake

```cmake
find_package(FlowEdge REQUIRED)
target_link_libraries(my_node PRIVATE FlowEdge::Core)
```

Source: `src/core/api/engine.h`. See [ADR 0003](../decisions/0003-api-boundary).
