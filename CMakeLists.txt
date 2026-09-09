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
option(FLOWEDGE_RELAY "Build the local real-time inference relay" OFF)
set(FLOWEDGE_RELAY_MAX_CONDITION_DIM "4096" CACHE STRING
    "Maximum condition width stored in a Relay message")
set(FLOWEDGE_RELAY_MAX_ACTION_DIM "1024" CACHE STRING
    "Maximum action width stored in a Relay message")
set(FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES "65536" CACHE STRING
    "Maximum opaque payload stored in a generic Relay job message")

# Core compute kernels. CPU auto-selects the host ISA; scalar is also explicit
# for compatibility testing and targets where deploying AVX2 is undesirable.
set(FLOWEDGE_BACKEND "cpu" CACHE STRING "Kernel backend: cpu, avx512, or scalar")
set_property(CACHE FLOWEDGE_BACKEND PROPERTY STRINGS cpu avx512 scalar)

# CORE
add_subdirectory(src/core)
if(FLOWEDGE_RELAY)
  add_subdirectory(src/relay)
endif()

# EXAMPLES
foreach(ex flow_sample mamba_forward external_flow_sample streaming_snapshot)
  add_executable(${ex} examples/${ex}.cc)
  target_link_libraries(${ex} PRIVATE FlowEdge::Core)
endforeach()
if(FLOWEDGE_RELAY)
  add_executable(relay_client_sample examples/relay_client_sample.cc)
  target_link_libraries(relay_client_sample PRIVATE FlowEdge::Relay)

  add_executable(cooperative_job_sample examples/cooperative_job_sample.cc)
  target_link_libraries(cooperative_job_sample PRIVATE FlowEdge::Relay)

  add_executable(routed_job_sample examples/routed_job_sample.cc)
  target_link_libraries(routed_job_sample PRIVATE FlowEdge::Relay)

  add_executable(mamba_relay_stream examples/mamba_relay_stream.cc)
  target_link_libraries(mamba_relay_stream PRIVATE FlowEdge::Relay)

  add_executable(action_delivery_sample examples/action_delivery_sample.cc)
  target_link_libraries(action_delivery_sample PRIVATE FlowEdge::Relay)
endif()

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
  target_link_libraries(flowedge_kernels_bench PRIVATE FlowEdge::Core benchmark::benchmark)

  add_executable(flowedge_engine_bench bench/engine_bench.cc)
  target_link_libraries(flowedge_engine_bench PRIVATE FlowEdge::Core benchmark::benchmark)

  add_executable(flowedge_latency_bench bench/latency_bench.cc)
  target_link_libraries(flowedge_latency_bench PRIVATE FlowEdge::Core)

  add_executable(flowedge_model_latency_bench bench/model_latency_bench.cc)
  target_link_libraries(flowedge_model_latency_bench PRIVATE FlowEdge::Core)

  add_executable(flowedge_threaded_matmul_bench bench/threaded_matmul_bench.cc)
  target_link_libraries(flowedge_threaded_matmul_bench PRIVATE FlowEdge::Core benchmark::benchmark_main Threads::Threads)

  if(FLOWEDGE_RELAY)
    add_executable(flowedge_relay_bench bench/relay_bench.cc)
    target_link_libraries(flowedge_relay_bench PRIVATE FlowEdge::Relay)

    add_executable(flowedge_relay_pool_bench bench/relay_pool_bench.cc)
    target_link_libraries(flowedge_relay_pool_bench PRIVATE FlowEdge::Relay)

    add_executable(flowedge_cooperative_job_bench bench/cooperative_job_bench.cc)
    target_link_libraries(flowedge_cooperative_job_bench PRIVATE FlowEdge::Relay)

    add_executable(flowedge_job_queue_bench bench/job_queue_bench.cc)
    target_link_libraries(flowedge_job_queue_bench PRIVATE FlowEdge::Relay)

    add_executable(flowedge_job_qos_bench bench/job_qos_bench.cc)
    target_link_libraries(flowedge_job_qos_bench PRIVATE FlowEdge::Relay)

    add_executable(flowedge_mamba_stream_bench bench/mamba_stream_bench.cc)
    target_link_libraries(flowedge_mamba_stream_bench PRIVATE FlowEdge::Relay)

    add_executable(flowedge_worker_drain_bench bench/worker_drain_bench.cc)
    target_link_libraries(flowedge_worker_drain_bench PRIVATE FlowEdge::Relay)

    add_executable(flowedge_action_delivery_bench bench/action_delivery_bench.cc)
    target_link_libraries(flowedge_action_delivery_bench PRIVATE FlowEdge::Relay)
  endif()
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
  add_executable(flowedge_tests test/tests.cc test/safetensors_tests.cc)
  target_link_libraries(flowedge_tests PRIVATE FlowEdge::Core GTest::gtest_main Threads::Threads)

  if(FLOWEDGE_RELAY)
    add_executable(flowedge_job_service_fixture test/job_service_fixture.cc)
    target_link_libraries(flowedge_job_service_fixture PRIVATE FlowEdge::Relay)

    add_executable(flowedge_relay_tests test/relay_tests.cc test/relay_process_tests.cc)
    target_link_libraries(flowedge_relay_tests PRIVATE FlowEdge::Relay GTest::gtest_main)
    target_compile_definitions(flowedge_relay_tests PRIVATE
      FLOWEDGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      FLOWEDGE_RELAYD_PATH="$<TARGET_FILE:flowedge-relayd>"
      FLOWEDGE_JOBD_PATH="$<TARGET_FILE:flowedge-jobd>"
      FLOWEDGE_JOB_SERVICE_FIXTURE_PATH="$<TARGET_FILE:flowedge_job_service_fixture>")
    add_dependencies(flowedge_relay_tests flowedge-relayd flowedge-jobd flowedge_job_service_fixture)
  endif()
  
  include(GoogleTest)
  gtest_discover_tests(flowedge_tests)
  if(FLOWEDGE_RELAY)
    gtest_discover_tests(flowedge_relay_tests)
    add_test(NAME RelayTools.ClientHelp COMMAND flowedge-relayctl --help)
    add_test(NAME RelayTools.JobClientHelp COMMAND flowedge-jobctl --help)
    add_test(NAME RelayTools.JobDaemonHelp COMMAND flowedge-jobd --help)
    add_test(NAME RelayTools.TraceHelp COMMAND flowedge-relay-trace --help)
  endif()
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
      # Statically link the C++ runtime; package winpthread beside the extension below.
      target_link_options(flowedge PRIVATE -static-libgcc -static-libstdc++ "-Wl,--allow-multiple-definition")
      target_link_libraries(flowedge PRIVATE winpthread)
    endif()
  endif()
  target_link_libraries(flowedge PRIVATE FlowEdge::Core)
  if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    find_file(FLOWEDGE_WINPTHREAD_DLL NAMES libwinpthread-1.dll HINTS $ENV{PATH})
    if(NOT FLOWEDGE_WINPTHREAD_DLL)
      message(FATAL_ERROR "libwinpthread-1.dll is required to package the MinGW Python extension")
    endif()
    install(FILES "${FLOWEDGE_WINPTHREAD_DLL}" DESTINATION .)
  endif()
  install(TARGETS flowedge LIBRARY DESTINATION . RUNTIME DESTINATION .) # wheel root: pip install .
endif()

if(NOT SKBUILD)
include(GNUInstallDirs)
set(FLOWEDGE_INSTALL_TARGETS flowedge_core flowedge_engine)
if(TARGET flowedge_relay)
  list(APPEND FLOWEDGE_INSTALL_TARGETS flowedge_relay)
endif()
install(TARGETS ${FLOWEDGE_INSTALL_TARGETS}
        EXPORT FlowEdgeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/core/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
if(TARGET flowedge_relay)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/relay/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/relay
          FILES_MATCHING PATTERN "*.h")
endif()

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
