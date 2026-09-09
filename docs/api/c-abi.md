# C ABI

The C ABI is the stable public boundary: opaque handles, C linkage, return-code errors, and
caller-owned buffers.

## Link

```cmake
find_package(FlowEdge REQUIRED)
target_link_libraries(my_node PRIVATE FlowEdge::Core)
```

## Core surface

```c
fe_engine* fe_engine_load(const char* path);
fe_engine* fe_engine_load_with_threads(const char* path, unsigned worker_threads);
fe_weights* fe_weights_load(const char* path);
fe_engine* fe_engine_create_from_weights(const fe_weights*, unsigned worker_threads);
fe_engine* fe_engine_create_from_weights_auto(const fe_weights*);
void fe_weights_free(fe_weights*);
void fe_engine_free(fe_engine*);

int fe_engine_run(fe_engine*, const int32_t* tokens, size_t n, float* out);
int fe_engine_sample(fe_engine*, const int32_t* tokens, size_t n,
                     const float* noise, size_t steps, int method, float* action);
int fe_engine_sample_condition(fe_engine*, const float* condition,
                               const float* noise, size_t steps,
                               int method, float* action);
int fe_engine_sample_diffusion(fe_engine*, const float* condition,
                               const float* noise, size_t steps,
                               int scheduler, uint64_t seed, float* action);
int fe_engine_diffusion_denoise(fe_engine*, const float* condition,
                                const float* sample, float timestep,
                                float* predicted_noise);

int fe_engine_step(fe_engine*, int32_t token, float* out);
void fe_engine_reset(fe_engine*);
size_t fe_engine_decode_state_bytes(const fe_engine*);
int fe_engine_export_decode_state(const fe_engine*, void* dst, size_t bytes);
int fe_engine_import_decode_state(fe_engine*, const void* src, size_t bytes);
const char* fe_engine_last_error(void);
```

## Resumable flow solving

```c
int fe_engine_flow_begin(fe_engine*, const float* condition,
                         const float* noise, size_t steps, int method);
int fe_engine_flow_advance(fe_engine*, size_t step_budget,
                           float* action, size_t* steps_remaining);
void fe_engine_cancel_before(fe_engine*, uint64_t generation);
```

Use the request/metadata variants when a scheduler needs model identity, timestamps, deadlines,
generation, or status reporting. See `src/core/api/engine.h` for the complete declarations.

## Rules

| Rule | Meaning |
|---|---|
| Errors | No exception crosses the ABI; `nullptr` or non-zero means failure; call `fe_engine_last_error()` |
| Solvers | `FE_SOLVER_EULER`, `FE_SOLVER_HEUN`, `FE_SOLVER_RK4` |
| Threads | `0` means caller-thread-only; otherwise use the requested worker count |
| Ownership | `fe_engine` owns mutable state; `fe_weights` owns immutable tensors and may be released after engine creation |
| Concurrency | One engine is one mutable lane; only cancellation is scheduler-thread safe |
| Buffers | Input/output memory is caller-owned and must be large enough for the reported dimensions |
| Snapshots | Versioned, checksummed, little-endian; incompatible or corrupt state is rejected before mutation |
| Profiles | Deployment profile pointers are borrowed until `fe_engine_free`; legacy checkpoints may return code `2` |

Prefill and token-conditioned sampling accept 1–512 tokens. Streaming `step` does not grow a
sequence buffer. Source: `src/core/api/engine.h` and `src/core/protocol/contracts.h`.
