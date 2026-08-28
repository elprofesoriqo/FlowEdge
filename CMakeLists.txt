cmake_minimum_required(VERSION 3.21)
project(FlowEdge)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FLOWEDGE_TESTS "Build the GoogleTest suite (dev/CI-only)" OFF)
option(FLOWEDGE_BENCH "Build the Google Benchmark microbenchmarks (dev-only)" OFF)
option(FLOWEDGE_PYTHON "Build the pybind11 Python module (dev/verification)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
  add_compile_options(-mavx2 -mfma -fno-math-errno)
endif()

if(MSVC)
  add_compile_options(/D__restrict__=__restrict)
endif()

# Compute kernels. CPU is the only production backend; AVX-512 is an explicit CPU variant.
set(FLOWEDGE_BACKEND "cpu" CACHE STRING "Kernel backend: cpu or avx512")
set_property(CACHE FLOWEDGE_BACKEND PROPERTY STRINGS cpu avx512)

if(FLOWEDGE_BACKEND STREQUAL "cpu")
  set(KERNEL_SRC "")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    list(APPEND KERNEL_SRC src/kernels/cpu/kernels_avx2.cc)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|aarch64|ARM64")
    list(APPEND KERNEL_SRC src/kernels/cpu/kernels_neon.cc)
  else()
    list(APPEND KERNEL_SRC src/kernels/cpu/kernels_scalar.cc)
  endif()
elseif(FLOWEDGE_BACKEND STREQUAL "avx512")
  set(KERNEL_SRC src/kernels/cpu/kernels_avx512.cc)
  add_compile_options(-mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mfma -fno-math-errno)
else()
  message(FATAL_ERROR "FLOWEDGE_BACKEND='${FLOWEDGE_BACKEND}' is not implemented; use cpu or avx512")
endif()

# ENGINE
add_library(flowedge_engine STATIC
  ${KERNEL_SRC}
  src/arena/thread_pool.cc
  src/loader/safetensors.cc
  src/runtime/engine_runtime.cc
  src/models/mamba/mamba.cc
  src/heads/flow/flow.cc
  src/api/engine.cc
)
set_target_properties(flowedge_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CheckIncludeFileCXX)
check_include_file_cxx(mdspan HAS_MDSPAN)
if(NOT HAS_MDSPAN)
  include(FetchContent)
  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG mdspan-0.6.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(mdspan)
  target_link_libraries(flowedge_engine PUBLIC std::mdspan)
endif()
target_link_libraries(flowedge_engine PUBLIC Threads::Threads)

target_include_directories(flowedge_engine PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> 
  $<INSTALL_INTERFACE:include>
)

# EXAMPLES
foreach(ex flow_sample mamba_forward)
  add_executable(${ex} examples/${ex}.cc)
  target_link_libraries(${ex} PRIVATE flowedge_engine)
endforeach()

# BENCHMARKS
if(FLOWEDGE_BENCH)
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googlebenchmark)

  foreach(t benchmark benchmark_main)
    if(TARGET ${t})
      target_compile_options(${t} PRIVATE -Wno-c2y-extensions)
    endif()
  endforeach()

  add_executable(flowedge_kernels_bench bench/kernels_bench.cc)
  target_link_libraries(flowedge_kernels_bench PRIVATE flowedge_engine benchmark::benchmark)

  add_executable(flowedge_engine_bench bench/engine_bench.cc)
  target_link_libraries(flowedge_engine_bench PRIVATE flowedge_engine benchmark::benchmark)

  add_executable(flowedge_latency_bench bench/latency_bench.cc)
  target_link_libraries(flowedge_latency_bench PRIVATE flowedge_engine)

  add_executable(flowedge_threaded_matmul_bench bench/threaded_matmul_bench.cc)
  target_link_libraries(flowedge_threaded_matmul_bench PRIVATE flowedge_engine benchmark::benchmark_main Threads::Threads)
endif()

# TESTS
if(FLOWEDGE_TESTS)
  enable_testing()
  include(FetchContent)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)

  find_package(Threads REQUIRED)
  add_executable(flowedge_tests test/tests.cc)
  target_link_libraries(flowedge_tests PRIVATE flowedge_engine GTest::gtest_main Threads::Threads)
  
  include(GoogleTest)
  gtest_discover_tests(flowedge_tests)
endif()

# PYTHON (Pybind11)
if(FLOWEDGE_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(flowedge python/flowedge_ext.cc)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(flowedge PRIVATE -pthread)
    if(WIN32)
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # removed -mstackrealign here
      endif()
      # static libstdc++ (resolves std::call_once) but keep winpthread dynamic
      target_link_options(flowedge PRIVATE -static-libgcc -static-libstdc++ "-Wl,--allow-multiple-definition")
      target_link_libraries(flowedge PRIVATE winpthread)
    endif()
  endif()
  target_link_libraries(flowedge PRIVATE flowedge_engine)
  install(TARGETS flowedge LIBRARY DESTINATION . RUNTIME DESTINATION .) # wheel root: pip install .
endif()

if(NOT SKBUILD)
include(GNUInstallDirs)
install(TARGETS flowedge_engine
        EXPORT FlowEdgeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

set(FLOWEDGE_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/FlowEdge)
install(EXPORT FlowEdgeTargets
        FILE FlowEdgeTargets.cmake
        NAMESPACE FlowEdge::
        DESTINATION ${FLOWEDGE_CMAKEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlowEdgeConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfig.cmake"
  INSTALL_DESTINATION ${FLOWEDGE_CMAKEDIR})
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfigVersion.cmake"
  VERSION 0.1.0
  COMPATIBILITY AnyNewerVersion)
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfigVersion.cmake"
        DESTINATION ${FLOWEDGE_CMAKEDIR})
endif()
