add_library(loader safetensors.cc)
target_include_directories(loader PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)
target_link_libraries(loader PUBLIC arena)
