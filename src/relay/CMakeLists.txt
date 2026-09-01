add_library(flowedge_relay STATIC
  client/relay_client.cc
  jobs/cooperative_job.cc
  jobs/job_registry.cc
  jobs/state_capsule.cc
  protocol/trace.cc
  shared_memory/shared_memory_ring.cc
  scheduler/edf_scheduler.cc
  telemetry/metrics.cc
  worker/head_worker.cc
  worker/head_worker_pool.cc
  worker/job_worker_pool.cc
  worker/worker_topology.cc
)
add_library(FlowEdge::Relay ALIAS flowedge_relay)

set_target_properties(flowedge_relay PROPERTIES
  EXPORT_NAME Relay
  OUTPUT_NAME flowedge_relay
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(flowedge_relay PUBLIC FlowEdge::Core Threads::Threads)
target_include_directories(flowedge_relay PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(flowedge_relay PUBLIC
  FLOWEDGE_RELAY_MAX_CONDITION_DIM=${FLOWEDGE_RELAY_MAX_CONDITION_DIM}
  FLOWEDGE_RELAY_MAX_ACTION_DIM=${FLOWEDGE_RELAY_MAX_ACTION_DIM}
  FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES=${FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES}
)
if(UNIX AND NOT APPLE)
  target_link_libraries(flowedge_relay PRIVATE rt)
endif()

add_executable(flowedge-relayd worker/relay_daemon.cc)
target_link_libraries(flowedge-relayd PRIVATE FlowEdge::Relay)

add_executable(flowedge-relayctl client/relay_control.cc)
target_link_libraries(flowedge-relayctl PRIVATE FlowEdge::Relay)

add_executable(flowedge-relay-trace tools/trace_tool.cc)
target_link_libraries(flowedge-relay-trace PRIVATE FlowEdge::Relay)

install(TARGETS flowedge-relayd flowedge-relayctl flowedge-relay-trace RUNTIME DESTINATION bin)
