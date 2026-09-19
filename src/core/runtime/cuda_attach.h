#pragma once

#include "runtime/environment.h"

#ifdef FLOWEDGE_CUDA
#include "kernels/cuda/kernels_cuda_api.h"

namespace fe {

[[nodiscard]] inline bool should_attach_cuda() noexcept
{
  return !environment_flag("FLOWEDGE_CUDA_FORCE_HOST") && cuda_ops::device_available();
}

} // namespace fe
#endif
