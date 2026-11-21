#pragma once

// Quantized GGUF matrix (Q8_0 / Q4_K / Q6_K, rows x K) expanded to row-major fp16, for cuBLAS prefill GEMMs.

#include <cuda_fp16.h>
#include <cstdint>

#include "ops.cuh"

namespace akd {

template <int type>
__global__ void pf_dequant_f16(const uint8_t* W, __half* out, int rows, int K) {
    const size_t n = static_cast<size_t>(rows) * K;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * 2;
    for (size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2; i < n; i += stride) {
        const size_t r = i / K;
        const int col = static_cast<int>(i - r * K);  // even, and K is a multiple of 32, so i + 1 is in the row
        float v0, v1;
        if (type == 0) {
            const uint8_t* blk = W + r * (K / 32) * 34 + (col >> 5) * 34;
            const float d = akd::qg_half(blk);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            v0 = d * static_cast<float>(qs[col & 31]);
            v1 = d * static_cast<float>(qs[(col & 31) + 1]);
        } else if (type == 1) {
            const uint8_t* blk = W + r * (K / 256) * 144 + (col >> 8) * 144;
            const float d = akd::qg_half(blk);
            const float dm = akd::qg_half(blk + 2);
            v0 = akd::q4k_at(blk + 4, blk + 16, d, dm, col & 255);
            v1 = akd::q4k_at(blk + 4, blk + 16, d, dm, (col & 255) + 1);
        } else {
            const uint8_t* blk = W + r * (K / 256) * 210 + (col >> 8) * 210;
            v0 = akd::q6k_at(blk, col & 255);
            v1 = akd::q6k_at(blk, (col & 255) + 1);
        }
        *reinterpret_cast<__half2*>(out + i) = __floats2half2_rn(v0, v1);
    }
}

}  // namespace akd
