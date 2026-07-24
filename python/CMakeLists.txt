# Python bindings (opt-in: -DFLOWEDGE_PYTHON=ON). 
# Builds the `flowedge` module, 
# used to verify C++ against PyTorch (scripts/verify_ulp.py).
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(flowedge flowedge_ext.cc)
target_compile_options(flowedge PRIVATE -pthread -mstackrealign)
target_link_options(flowedge PRIVATE -pthread -static -static-libgcc -static-libstdc++)
target_link_libraries(flowedge PRIVATE api)
