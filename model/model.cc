#include "model.h"

#include "cpu/linear.h"

void Model::execute()
{
  std::shared_ptr<Linear> linear = std::make_shared<CPULinear>();
  linear->compute();
}