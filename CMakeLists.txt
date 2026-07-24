cmake_minimum_required(VERSION 3.21)
project(FlowEdge)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FLOWEDGE_TESTS "Build the GoogleTest suite (dev/CI-only)" OFF)
option(FLOWEDGE_BENCH "Build the Google Benchmark microbenchmarks (dev-only)" OFF)

# Enable AVX2+FMA on x86
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
  add_compile_options(-mavx2 -mfma -fno-math-errno)
endif()

add_subdirectory(src)
add_subdirectory(examples)

if(FLOWEDGE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
if(FLOWEDGE_BENCH)
  add_subdirectory(bench)
endif()
