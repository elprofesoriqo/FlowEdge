add_library(loader safetensors.cc)
target_include_directories(loader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(loader PRIVATE arena)
