add_library(flowedge_relay STATIC
  protocol/trace.cc
  shared_memory/shared_memory_ring.cc
  scheduler/edf_scheduler.cc
  worker/head_worker.cc
)
add_library(FlowEdge::Relay ALIAS flowedge_relay)

set_target_properties(flowedge_relay PROPERTIES
  EXPORT_NAME Relay
  OUTPUT_NAME flowedge_relay
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(flowedge_relay PUBLIC FlowEdge::Core)
target_include_directories(flowedge_relay PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(flowedge_relay PUBLIC
  FLOWEDGE_RELAY_MAX_CONDITION_DIM=${FLOWEDGE_RELAY_MAX_CONDITION_DIM}
  FLOWEDGE_RELAY_MAX_ACTION_DIM=${FLOWEDGE_RELAY_MAX_ACTION_DIM}
)
if(UNIX AND NOT APPLE)
  target_link_libraries(flowedge_relay PRIVATE rt)
endif()

add_executable(flowedge-relayd worker/relay_daemon.cc)
target_link_libraries(flowedge-relayd PRIVATE FlowEdge::Relay)
install(TARGETS flowedge-relayd RUNTIME DESTINATION bin)
