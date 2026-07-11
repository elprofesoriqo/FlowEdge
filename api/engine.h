#ifndef ENGINE_H
#define ENGINE_H

#include "../model/model.h"

#include <memory>

class Engine
{
public:
  Engine();

  void run();

private:
  std::shared_ptr<Model> model{std::make_shared<Model>()};
};

#endif // ENGINE_H