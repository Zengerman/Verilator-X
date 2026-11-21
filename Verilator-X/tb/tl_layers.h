// .h
#pragma once
#include <cstdint>
void tl_max_pool_2x2(const uint8_t* in, uint8_t* out, int in_size, int ch);
void tl_residual_add (const uint8_t* a, const uint8_t* b, uint8_t* y, int size, int ch);
