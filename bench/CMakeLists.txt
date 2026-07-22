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

add_executable(flowedge_kernels_bench kernels_bench.cc)
target_link_libraries(flowedge_kernels_bench PRIVATE cpu_model benchmark::benchmark)
target_include_directories(flowedge_kernels_bench PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(flowedge_engine_bench engine_bench.cc)
target_link_libraries(flowedge_engine_bench PRIVATE api benchmark::benchmark)
