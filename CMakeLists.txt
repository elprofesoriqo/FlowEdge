cmake_minimum_required(VERSION 3.21)
project(FlowEdge)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable AVX2+FMA on x86
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
  add_compile_options(-mavx2 -mfma -fno-math-errno)
endif()

add_subdirectory(src)
add_subdirectory(api)
add_subdirectory(model)
