#ifndef LINEAR_H
#define LINEAR_H

#include <memory>

class Linear {
public:
    virtual ~Linear() = default;

    virtual void compute() = 0;
};

#endif // LINEAR_H