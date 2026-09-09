set(FLOWEDGE_CORE_KERNEL_SOURCE)

if(FLOWEDGE_BACKEND STREQUAL "cpu")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    set(FLOWEDGE_CORE_KERNEL_SOURCE kernels/cpu/kernels_avx2.cc)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|aarch64|ARM64")
    set(FLOWEDGE_CORE_KERNEL_SOURCE kernels/cpu/kernels_neon.cc)
  else()
    set(FLOWEDGE_CORE_KERNEL_SOURCE kernels/cpu/kernels_scalar.cc)
  endif()
elseif(FLOWEDGE_BACKEND STREQUAL "avx512")
  set(FLOWEDGE_CORE_KERNEL_SOURCE kernels/cpu/kernels_avx512.cc)
elseif(FLOWEDGE_BACKEND STREQUAL "scalar")
  set(FLOWEDGE_CORE_KERNEL_SOURCE kernels/cpu/kernels_scalar.cc)
else()
  message(FATAL_ERROR
    "FLOWEDGE_BACKEND='${FLOWEDGE_BACKEND}' is not implemented; use cpu, avx512, or scalar")
endif()

add_library(flowedge_core STATIC
  ${FLOWEDGE_CORE_KERNEL_SOURCE}
  arena/thread_pool.cc
  loader/safetensors.cc
  protocol/deployment_profile.cc
  protocol/model_identity.cc
  protocol/snapshot.cc
  runtime/engine_runtime.cc
  models/mamba/mamba.cc
  heads/flow/flow.cc
  api/engine.cc
)
add_library(FlowEdge::Core ALIAS flowedge_core)

# Build-tree compatibility for existing users. Installed packages expose both
# FlowEdge::Core and the legacy FlowEdge::flowedge_engine forwarding target.
add_library(flowedge_engine INTERFACE)
target_link_libraries(flowedge_engine INTERFACE flowedge_core)

set_target_properties(flowedge_core PROPERTIES
  EXPORT_NAME Core
  OUTPUT_NAME flowedge_engine
  POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86"
   AND NOT FLOWEDGE_BACKEND STREQUAL "scalar")
  target_compile_options(flowedge_core PRIVATE -mavx2 -mfma -fno-math-errno)
endif()
if(FLOWEDGE_BACKEND STREQUAL "avx512")
  target_compile_options(flowedge_core PRIVATE
    -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mfma -fno-math-errno)
endif()
if(MSVC)
  target_compile_definitions(flowedge_core PRIVATE __restrict__=__restrict)
endif()

include(CheckIncludeFileCXX)
check_include_file_cxx(mdspan HAS_MDSPAN)
set(FLOWEDGE_NEEDS_MDSPAN OFF PARENT_SCOPE)
if(NOT HAS_MDSPAN)
  include(FetchContent)
  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG mdspan-0.6.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(mdspan)
  target_link_libraries(flowedge_core PUBLIC std::mdspan)
  set(FLOWEDGE_NEEDS_MDSPAN ON PARENT_SCOPE)
endif()

target_link_libraries(flowedge_core PUBLIC Threads::Threads)
target_include_directories(flowedge_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
