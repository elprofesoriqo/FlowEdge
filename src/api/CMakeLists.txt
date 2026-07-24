add_library(api STATIC engine.cc engine.h)
target_link_libraries(api PUBLIC mamba)
target_include_directories(api PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include>)
