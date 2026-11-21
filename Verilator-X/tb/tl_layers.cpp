// .cpp
#include "tl_layers.h"
#include <algorithm>
void tl_max_pool_2x2(const uint8_t* in, uint8_t* out, int in_size, int ch) {
    int os = in_size / 2;
    for (int i = 0; i < os; ++i)
        for (int j = 0; j < os; ++j)
            for (int c = 0; c < ch; ++c) {
                auto at = [&](int a, int b){ return ((a*in_size + b)*ch) + c; };
                uint8_t m = in[at(2*i,2*j)];
                m = std::max(m, in[at(2*i+1,2*j)]);
                m = std::max(m, in[at(2*i,2*j+1)]);
                m = std::max(m, in[at(2*i+1,2*j+1)]);
                out[((i*os + j)*ch) + c] = m;
            }
}
void tl_residual_add(const uint8_t* a, const uint8_t* b, uint8_t* y, int size, int ch) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            for (int c = 0; c < ch; ++c) {
                int idx = ((i*size + j)*ch) + c;
                y[idx] = (a[idx] + b[idx]) & 0xFF;
            }
}
