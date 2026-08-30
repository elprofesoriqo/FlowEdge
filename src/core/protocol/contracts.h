#ifndef FE_PROTOCOL_CONTRACTS_H
#define FE_PROTOCOL_CONTRACTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum
{
  FE_PROTOCOL_VERSION = 1,
  FE_MODEL_DIGEST_BYTES = 16,
};

enum
{
  FE_ARCH_UNKNOWN = 0,
  FE_ARCH_MAMBA = 1,
  FE_ARCH_FLOW_HEAD = 2,
  FE_ARCH_MAMBA_FLOW = 3,
};

enum
{
  FE_PRECISION_UNKNOWN = 0,
  FE_PRECISION_F32 = 1,
  FE_PRECISION_BF16 = 2,
  FE_PRECISION_MIXED = 3,
};

enum
{
  FE_ACTION_RUNNING = 0,
  FE_ACTION_COMPLETE = 1,
  FE_ACTION_CANCELLED = 2,
  FE_ACTION_FAILED = 3,
};

typedef struct fe_model_digest
{
  uint8_t bytes[FE_MODEL_DIGEST_BYTES];
} fe_model_digest;

typedef struct fe_model_metadata
{
  uint32_t struct_size;
  uint32_t protocol_version;
  uint32_t architecture;
  uint32_t precision;
  fe_model_digest model_digest;
  uint64_t d_model;
  uint64_t n_layers;
  uint64_t d_inner;
  uint64_t d_state;
  uint64_t d_conv;
  uint64_t action_dim;
  uint64_t condition_dim;
  uint64_t decode_snapshot_bytes;
} fe_model_metadata;

typedef struct fe_condition_metadata
{
  uint32_t struct_size;
  uint32_t protocol_version;
  uint32_t solver;
  uint32_t reserved;
  fe_model_digest model_digest;
  uint64_t timestamp_ns;
  uint64_t deadline_ns;
  uint64_t generation;
  uint64_t condition_dim;
  uint64_t action_dim;
  uint64_t solver_steps;
  uint64_t remaining_nfe;
} fe_condition_metadata;

typedef struct fe_action_metadata
{
  uint32_t struct_size;
  uint32_t protocol_version;
  uint32_t solver;
  uint32_t status;
  fe_model_digest model_digest;
  uint64_t timestamp_ns;
  uint64_t deadline_ns;
  uint64_t generation;
  uint64_t condition_dim;
  uint64_t action_dim;
  uint64_t remaining_nfe;
} fe_action_metadata;

#ifdef __cplusplus
}

static_assert(sizeof(fe_model_digest) == 16);
static_assert(sizeof(fe_model_metadata) == 96);
static_assert(sizeof(fe_condition_metadata) == 88);
static_assert(sizeof(fe_action_metadata) == 80);
#endif

#endif // FE_PROTOCOL_CONTRACTS_H
