add_library(api STATIC engine.cc engine.h)
target_link_libraries(api PUBLIC model)
target_include_directories(api PUBLIC ${CMAKE_SOURCE_DIR})
