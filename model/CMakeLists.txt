add_library(model STATIC mamba.cc mamba.h)
target_link_libraries(model PUBLIC cpu_model loader)
add_subdirectory(cpu)
