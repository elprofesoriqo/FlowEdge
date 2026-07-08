#ifndef MODEL_H
#define MODEL_H

#include <memory>

class Attention;
class Linear;

class Model {
public:
    void execute();

private:
    std::shared_ptr<Attention> attention;
    std::shared_ptr<Linear> linear;
};

#endif // MODEL_H