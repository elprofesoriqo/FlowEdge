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

// Diffusion Policy action head; obs_cond (perception output) is supplied by the caller.
typedef struct FeDenoiser fe_denoiser;

fe_denoiser* fe_denoiser_load(const char* path);

// Action dimension, or 0 if the checkpoint has no denoiser.
size_t fe_denoiser_action_dim(const fe_denoiser* denoiser);

// Length of the observation-conditioning vector expected by fe_denoiser_sample.
size_t fe_denoiser_obs_cond_dim(const fe_denoiser* denoiser);

// Sample an action trajectory: DDIM(noise, obs_cond) over `steps` denoising steps.
// obs_cond is obs_cond_dim floats; noise and action are horizon*action_dim floats.
int fe_denoiser_sample(fe_denoiser* denoiser, const float* obs_cond, const float* noise,
                       size_t horizon, size_t steps, float* action);

void fe_denoiser_free(fe_denoiser* denoiser);

#ifdef __cplusplus
}
#endif

#endif // FE_ENGINE_H
