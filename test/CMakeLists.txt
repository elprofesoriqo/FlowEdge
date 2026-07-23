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
add_executable(flowedge_tests tests.cc)
target_link_libraries(flowedge_tests PRIVATE cpu_model flow GTest::gtest_main Threads::Threads)
target_include_directories(flowedge_tests PRIVATE ${CMAKE_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(flowedge_tests)
