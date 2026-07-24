# FlowEdge engine. Dependency order: shared primitives -> models -> C-ABI.
add_subdirectory(arena)   # fixed-state bump allocator
add_subdirectory(loader)  # .safetensors mmap loader
add_subdirectory(kernels) # backend-selectable compute (cpu today; gpu/npu later)
add_subdirectory(models)  # action-head models (mamba)
add_subdirectory(api)     # C-ABI over the models

install(TARGETS arena loader kernels mamba api
        EXPORT FlowEdgeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
