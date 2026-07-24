# FlowEdge engine. Dependency order: shared primitives -> models -> C-ABI.
add_subdirectory(arena)   # fixed-state bump allocator
add_subdirectory(loader)  # .safetensors mmap loader
add_subdirectory(kernels) # backend-selectable compute (cpu today; gpu/npu later)
add_subdirectory(models)  # mamba + diffusion action heads
add_subdirectory(api)     # C-ABI over the models
