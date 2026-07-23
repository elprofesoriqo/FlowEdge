add_library(flow STATIC flow.cc flow.h)
target_link_libraries(flow PUBLIC cpu_model loader)
