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

void fe_engine_free(fe_engine* engine);

#ifdef __cplusplus
}
#endif

#endif // FE_ENGINE_H
