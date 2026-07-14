add_library(model
  model.cc
  model.h
  linear.h
)
target_link_libraries(model PRIVATE cpu_model)
add_subdirectory(cpu)