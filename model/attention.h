#ifndef ATTENTION_H
#define ATTENTION_H

#include <memory>

class Attention {
public:
    virtual ~Attention() = default;

    virtual void compute() = 0;
};

#endif // ATTENTION_H