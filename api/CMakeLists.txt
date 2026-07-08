add_library(api STATIC engine.cc engine.h)
target_link_libraries(api PRIVATE model)