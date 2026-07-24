add_library(arena INTERFACE)
target_include_directories(arena INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include>)
