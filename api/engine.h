#ifndef FE_ENGINE_H
#define FE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FeEngine fe_engine;

fe_engine* fe_engine_load(const char* path);

// Model dimensions
void fe_engine_dims(const fe_engine* engine, size_t* d_model, size_t* n_layers);

int fe_engine_run(fe_engine* engine, const int32_t* tokens, size_t seq_len, float* out);

// streaming decode: advance one token against the engine's persistent SSM state, out is d_model
int fe_engine_step(fe_engine* engine, int32_t token, float* out);

// begin a fresh sequence
void fe_engine_reset(fe_engine* engine);

// Flow-head action dimension, or 0 if the checkpoint has no flow head.
size_t fe_engine_action_dim(const fe_engine* engine);

// Sample an action: Mamba(tokens) -> last hidden = cond -> ODE(noise) -> action.
// noise and action are action_dim floats
// method 0=Euler, 1=Heun
int fe_engine_sample(fe_engine* engine, const int32_t* tokens, size_t seq_len, const float* noise,
                     size_t steps, int method, float* action);

void fe_engine_free(fe_engine* engine);

#ifdef __cplusplus
}
#endif

#endif // FE_ENGINE_H
