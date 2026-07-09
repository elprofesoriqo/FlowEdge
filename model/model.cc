#include "model.h"
#include "cpu/attention.h"
#include "cpu/linear.h"

void Model::execute() {
    std::shared_ptr<Attention> attention = std::make_shared<CPUAttention>();
    std::shared_ptr<Linear> linear = std::make_shared<CPULinear>();

    attention->compute();
    linear->compute();
}