#ifndef CPU_LINEAR_H
#define CPU_LINEAR_H

#include "../linear.h"

class CPULinear : public Linear {
public:
    void compute() override;
};

#endif // CPU_LINEAR_H