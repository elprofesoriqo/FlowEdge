#include "engine.h"
#include "../model/model.h"
#include <iostream>

Engine::Engine()
    : model(std::make_shared<Model>()) {}

void Engine::run() {
    std::cout << "Running engine..." << std::endl;
    model->execute();
}