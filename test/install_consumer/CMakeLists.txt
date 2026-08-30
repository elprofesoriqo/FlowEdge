cmake_minimum_required(VERSION 3.21)
project(FlowEdgeInstallConsumer LANGUAGES CXX)

find_package(FlowEdge CONFIG REQUIRED)

add_executable(flowedge_install_consumer main.cc)
target_compile_features(flowedge_install_consumer PRIVATE cxx_std_23)
target_link_libraries(flowedge_install_consumer PRIVATE FlowEdge::Core FlowEdge::Relay)
