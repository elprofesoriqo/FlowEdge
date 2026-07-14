#include "engine.h"

#include "../model/model.h"

#include <iostream>

Engine::Engine() = default;

void Engine::run()
{
  std::cout << "Running engine...\n";
  model->execute();
}