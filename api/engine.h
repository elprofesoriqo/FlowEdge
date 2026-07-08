#ifndef ENGINE_H
#define ENGINE_H

#include <memory>
#include "../model/model.h"

class Engine {
public:
    Engine();

    void run();

private:
    std::shared_ptr<Model> model;
};

#endif // ENGINE_H