#ifndef FE_ENGINE_H
#define FE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FeEngine fe_engine;

enum
{
  FE_SOLVER_EULER = 0,
  FE_SOLVER_HEUN = 1,
  FE_SOLVER_RK4 = 2,
};

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
 * @brief Load a model with an exact number of background worker threads.
 *
 * Passing 0 selects caller-thread-only execution. Values above 8 are rejected.
 * fe_engine_load() instead uses FLOWEDGE_THREADS when set, otherwise a
 * bandwidth-aware automatic default.
 */
fe_engine* fe_engine_load_with_threads(const char* path, unsigned worker_threads);

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
 * @brief Get the condition-vector dimension accepted by the flow head.
 * @param engine The engine instance.
 * @return The condition dimension, or 0 if the checkpoint lacks a flow head.
 */
size_t fe_engine_condition_dim(const fe_engine* engine);

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
 * @param method One of FE_SOLVER_EULER, FE_SOLVER_HEUN, or FE_SOLVER_RK4.
 * @param action Output buffer of size [action_dim] for the final action.
 * @return 0 on success, non-zero on error.
 */
int fe_engine_sample(fe_engine* engine, const int32_t* tokens, size_t seq_len, const float* noise,
                     size_t steps, int method, float* action);

/**
 * @brief Sample directly from an externally produced condition vector.
 *
 * This bypasses the built-in backbone and lets a vision encoder, VLA, or other
 * runtime reuse FlowEdge's deterministic action head without copying through a
 * token representation.
 *
 * @param engine The engine instance.
 * @param condition Input array of size [condition_dim].
 * @param noise Initial Gaussian noise array of size [action_dim].
 * @param steps Number of ODE solver steps.
 * @param method One of FE_SOLVER_EULER, FE_SOLVER_HEUN, or FE_SOLVER_RK4.
 * @param action Output buffer of size [action_dim].
 * @return 0 on success, non-zero on error.
 */
int fe_engine_sample_condition(fe_engine* engine, const float* condition, const float* noise,
                               size_t steps, int method, float* action);

/**
 * @brief Begin an allocation-free, resumable flow solve from an external condition.
 *
 * Only one resumable solve may be active per engine. Starting another replaces
 * the previous solve. Use fe_engine_flow_advance() to run bounded work units.
 *
 * @return 0 on success, non-zero on error.
 */
int fe_engine_flow_begin(fe_engine* engine, const float* condition, const float* noise,
                         size_t steps, int method);

/**
 * @brief Advance a resumable solve by at most `step_budget` complete ODE steps.
 *
 * The current solver state is copied to `action`; publish it as a final action
 * when `steps_remaining` reaches zero.
 *
 * @param engine The engine instance.
 * @param step_budget Maximum number of ODE steps to execute in this call.
 * @param action Output buffer of size [action_dim].
 * @param steps_remaining Optional output for the number of unfinished steps.
 * @return 0 on success, non-zero on error.
 */
int fe_engine_flow_advance(fe_engine* engine, size_t step_budget, float* action,
                           size_t* steps_remaining);

/**
 * @brief Return the byte size of the model's streaming decode state.
 *
 * A snapshot is valid only for another engine loaded from the same checkpoint.
 */
size_t fe_engine_decode_state_bytes(const fe_engine* engine);

/**
 * @brief Copy the streaming decode state into caller-owned storage.
 * @param destination Buffer with at least fe_engine_decode_state_bytes() bytes.
 */
int fe_engine_export_decode_state(const fe_engine* engine, void* destination, size_t bytes);

/**
 * @brief Restore a streaming state previously exported from the same model.
 * @param source Snapshot buffer of exactly fe_engine_decode_state_bytes() bytes.
 */
int fe_engine_import_decode_state(fe_engine* engine, const void* source, size_t bytes);

/**
 * @brief Free the engine resources.
 * @param engine The engine instance to destroy.
 */
void fe_engine_free(fe_engine* engine);

#ifdef __cplusplus
}
#endif

#endif // FE_ENGINE_H
