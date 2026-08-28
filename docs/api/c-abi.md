# C-ABI

The only public surface. C linkage, opaque handle.

```c
fe_engine* fe_engine_load(const char* path);
void       fe_engine_free(fe_engine* e);

void   fe_engine_dims(const fe_engine*, size_t* d_model, size_t* n_layers);
size_t fe_engine_action_dim(const fe_engine*);

int fe_engine_run(fe_engine*, const int32_t* tokens, size_t n, float* out);
int fe_engine_sample(fe_engine*, const int32_t* tokens, size_t n,
                     const float* noise, size_t steps, int method, float* action);

int  fe_engine_step(fe_engine*, int32_t token, float* out);
void fe_engine_reset(fe_engine*);

const char* fe_engine_last_error(void);
```

Rules:

- Everything runtime-relevant is a parameter. Nothing is baked in.
- The handle owns the whole runtime. Free it with `fe_engine_free`.
- No exception crosses the boundary. Errors return `nullptr` or a non-zero code.
- `method` is 0 for Euler, 1 for Heun, 2 for RK4.
- Prefer the named constants `FE_SOLVER_EULER`, `FE_SOLVER_HEUN`, and `FE_SOLVER_RK4` from
  `engine.h` instead of literal method values. Unknown values fail with a non-zero return code.

## Link with CMake

```cmake
find_package(FlowEdge REQUIRED)
target_link_libraries(my_node PRIVATE FlowEdge::flowedge_engine)
```

Source: `src/api/engine.h`. See [ADR 0003](../decisions/0003-api-boundary).
