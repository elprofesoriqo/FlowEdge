# Compute kernels. The interface (kernels.h) is backend-agnostic; each backend supplies the
# implementation. Select with -DFLOWEDGE_BACKEND=cpu (default). Drop in gpu/ (CUDA) or npu/ later.
set(FLOWEDGE_BACKEND "cpu" CACHE STRING "Kernel backend: cpu (gpu/npu to come)")

if(FLOWEDGE_BACKEND STREQUAL "cpu")
  add_library(kernels STATIC cpu/kernels.cc kernels.h)
else()
  message(FATAL_ERROR "FLOWEDGE_BACKEND='${FLOWEDGE_BACKEND}' has no implementation yet")
endif()

target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/src)
