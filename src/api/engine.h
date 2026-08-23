#ifndef FE_ENGINE_H
#define FE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FeEngine fe_engine;

/**
 * @brief Returns the last error message encountered by the engine on the current thread.
 * @return A null-terminated string describing the error, or an empty string if no error occurred.
 */
const char* fe_engine_last_error(void);

/**
 * @brief Load a FlowEdge model from a .safetensors file.
 * @param path Path to the .safetensors file.
 * @return Pointer to the initialized engine, or NULL on failure. Check
 * fe_engine_last_error() on failure.
 */
fe_engine* fe_engine_load(const char* path);

/**
 * @brief Get the underlying backbone model dimensions.
 * @param engine The engine instance.
 * @param d_model Output pointer for the model dimension (can be NULL).
 * @param n_layers Output pointer for the number of layers (can be NULL).
 */
void fe_engine_dims(const fe_engine* engine, size_t* d_model, size_t* n_layers);

/**
 * @brief Process a sequence of tokens in batch mode (prefill).
 * @param engine The engine instance.
 * @param tokens Array of input token IDs.
 * @param seq_len Length of the token array.
 * @param out Output buffer of size [seq_len * d_model] for the hidden states.
 * @return 0 on success, non-zero on error.
 */
int fe_engine_run(fe_engine* engine, const int32_t* tokens, size_t seq_len, float* out);

/**
 * @brief Advance the streaming decode state by a single token.
 * @param engine The engine instance.
 * @param token The input token ID.
 * @param out Output buffer of size [d_model] for the current hidden state.
 * @return 0 on success, non-zero on error.
 */
int fe_engine_step(fe_engine* engine, int32_t token, float* out);

/**
 * @brief Reset the internal streaming state for a new sequence.
 * @param engine The engine instance.
 */
void fe_engine_reset(fe_engine* engine);

/**
 * @brief Get the number of background worker threads in the engine.
 * @param engine The engine instance.
 * @return The number of worker threads (0 if single-threaded).
 */
unsigned fe_engine_thread_count(const fe_engine* engine);

/**
 * @brief Get the action dimension of the flow-matching head.
 * @param engine The engine instance.
 * @return The action dimension, or 0 if the checkpoint lacks a flow head.
 */
size_t fe_engine_action_dim(const fe_engine* engine);

/**
 * @brief Sample an action trajectory from the flow-matching head.
 *
 * This function processes the input tokens to generate a conditioning context,
 * then solves the ODE from the provided noise to the final action.
 *
 * @param engine The engine instance.
 * @param tokens Array of conditioning token IDs (prefix).
 * @param seq_len Length of the token array.
 * @param noise Initial Gaussian noise array of size [action_dim].
 * @param steps Number of ODE solver steps.
 * @param method The ODE solver to use (0 = Euler, 1 = Heun, 2 = RK4).
 * @param action Output buffer of size [action_dim] for the final action.
 * @return 0 on success, non-zero on error.
 */
int fe_engine_sample(fe_engine* engine, const int32_t* tokens, size_t seq_len, const float* noise,
                     size_t steps, int method, float* action);

/**
 * @brief Free the engine resources.
 * @param engine The engine instance to destroy.
 */
void fe_engine_free(fe_engine* engine);

#ifdef __cplusplus
}
#endif

#endif // FE_ENGINE_H
