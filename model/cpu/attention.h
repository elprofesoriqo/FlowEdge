#ifndef CPU_ATTENTION_H
#define CPU_ATTENTION_H

#include "../attention.h"

class CPUAttention : public Attention {
public:
    void compute() override;
};

#endif // CPU_ATTENTION_H