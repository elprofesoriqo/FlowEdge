# Mamba SSM backbone + its flow-matching action head
add_library(mamba STATIC mamba.cc mamba.h flow.cc flow.h)
target_link_libraries(mamba PUBLIC kernels loader)
