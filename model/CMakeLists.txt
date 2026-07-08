

add_library(model STATIC model.cc model.h attention.h linear.h)
target_link_libraries(model PRIVATE cpu_model)
add_subdirectory(cpu)