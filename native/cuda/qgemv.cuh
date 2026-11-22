#pragma once

// Bandwidth-oriented quantized GEMV for batch-1 decode.
//
// The activation vector is quantized once to Q8_1-style blocks (32 int8 values plus one float scale).
// Weight blocks are then dotted with __dp4a, four int8 products per instruction, instead of decoding every
// weight to float. Each warp owns one output row, and its lanes split a row's super-blocks so that one warp
// load covers contiguous weight bytes.
//
// Layout of x after quantization:  xq[K] int8, xd[K / 32] float, x[i] ~= xd[i / 32] * xq[i].
// xq must be 16-byte aligned. Q4_K rows must start 16-byte aligned (GGUF tensors are 32-aligned and 144 is a
// multiple of 16). Q6_K and Q8_0 blocks are only 2-byte aligned and use 16-bit loads.

#include <cuda_fp16.h>
#include <cstdint>

namespace akd {

__device__ __forceinline__ float qg_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float qg_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float qg_half(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// Little-endian 32-bit word from a 2-byte-aligned address.
__device__ __forceinline__ int qg_ld_b2(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return static_cast<int>(static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16));
}

// Block-wide: quantize x[0..K) into xq / xd. K must be a multiple of 32.
__device__ __forceinline__ void quantize_q8_1_n(const float* x, int8_t* xq, float* xd, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    for (int base = warp * 32; base < K; base += nwarps * 32) {
        const float v = x[base + lane];
        const float amax = qg_warp_max(fabsf(v));
        const float d = amax / 127.f;
        const int q = amax > 0.f ? __float2int_rn(v / d) : 0;
        xq[base + lane] = static_cast<int8_t>(q);
        if (lane == 0) xd[base >> 5] = d;
    }
    __syncthreads();
}

// Grid-wide quantize: one warp per 32-value block. Launch with ceil(K / 32 / warps_per_block) blocks.
static __global__ void quantize_q8_1_grid(const float* x, int8_t* xq, float* xd, int K) {
    const int lane = threadIdx.x & 31;
    const int base = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 32;
    if (base >= K) return;
    const float v = x[base + lane];
    const float amax = qg_warp_max(fabsf(v));
    const float d = amax / 127.f;
    const int q = amax > 0.f ? __float2int_rn(v / d) : 0;
    xq[base + lane] = static_cast<int8_t>(q);
    if (lane == 0) xd[base >> 5] = d;
}

// ---- Q4_K: 144-byte super-block, 256 weights. 8 lanes per super-block, 16 weight bytes per lane. ----
// q4k_lane decodes what a lane needs from the weights once; q4k_col dots it with one activation column.
struct Q4kLane {
    int4 w;
    int sc0, m0, sc1, m1;
    float2 dm;
    int xoff;  // byte offset of this lane's activations inside the super-block
    int is;    // first of the two 32-value sub-blocks this lane covers
};

__device__ __forceinline__ Q4kLane q4k_lane(const uint8_t* blk, int t) {
    Q4kLane L;
    const int j = t >> 1;           // 64-weight group
    const int off = (t & 1) << 4;   // first or second 16 bytes of the group's 32 bytes
    L.w = *reinterpret_cast<const int4*>(blk + 16 + (t << 4));
    L.xoff = j * 64 + off;
    // 6-bit scales and mins for sub-blocks 2j and 2j+1 (same packing as get_scale_min_k4).
    const uint8_t* s = blk + 4;
    const int is = 2 * j;
    L.is = is;
    if (is < 4) {
        L.sc0 = s[is] & 63;
        L.m0 = s[is + 4] & 63;
        L.sc1 = s[is + 1] & 63;
        L.m1 = s[is + 5] & 63;
    } else {
        L.sc0 = (s[is + 4] & 0xF) | ((s[is - 4] >> 6) << 4);
        L.m0 = (s[is + 4] >> 4) | ((s[is] >> 6) << 4);
        L.sc1 = (s[is + 5] & 0xF) | ((s[is - 3] >> 6) << 4);
        L.m1 = (s[is + 5] >> 4) | ((s[is + 1] >> 6) << 4);
    }
    L.dm = __half22float2(*reinterpret_cast<const __half2*>(blk));
    return L;
}

// What a lane reads from one activation column for one super-block: its 2 x 16 int8 values, their sums (for the
// min term) and the two sub-block scales. It depends only on the lane's position, not on the weight row, so a lane
// that computes several rows loads it once (qgemv_dp4a_rows_mc2).
struct Q4kX {
    int4 xa, xb;
    int sl, sh;
    float x0, x1;
};

// xq, xd point at this super-block's activations (256 int8, 8 scales).
__device__ __forceinline__ Q4kX q4k_x(const Q4kLane& L, const int8_t* xq, const float* xd) {
    Q4kX X;
    X.xa = *reinterpret_cast<const int4*>(xq + L.xoff);
    X.xb = *reinterpret_cast<const int4*>(xq + L.xoff + 32);
    const int ones = 0x01010101;
    int sl = 0, sh = 0;
    sl = __dp4a(ones, X.xa.x, sl);
    sl = __dp4a(ones, X.xa.y, sl);
    sl = __dp4a(ones, X.xa.z, sl);
    sl = __dp4a(ones, X.xa.w, sl);
    sh = __dp4a(ones, X.xb.x, sh);
    sh = __dp4a(ones, X.xb.y, sh);
    sh = __dp4a(ones, X.xb.z, sh);
    sh = __dp4a(ones, X.xb.w, sh);
    X.sl = sl;
    X.sh = sh;
    X.x0 = xd[L.is];
    X.x1 = xd[L.is + 1];
    return X;
}

__device__ __forceinline__ float q4k_dot(const Q4kLane& L, const Q4kX& X) {
    int dl = 0, dh = 0;
    dl = __dp4a(L.w.x & 0x0F0F0F0F, X.xa.x, dl);
    dl = __dp4a(L.w.y & 0x0F0F0F0F, X.xa.y, dl);
    dl = __dp4a(L.w.z & 0x0F0F0F0F, X.xa.z, dl);
    dl = __dp4a(L.w.w & 0x0F0F0F0F, X.xa.w, dl);
    dh = __dp4a((L.w.x >> 4) & 0x0F0F0F0F, X.xb.x, dh);
    dh = __dp4a((L.w.y >> 4) & 0x0F0F0F0F, X.xb.y, dh);
    dh = __dp4a((L.w.z >> 4) & 0x0F0F0F0F, X.xb.z, dh);
    dh = __dp4a((L.w.w >> 4) & 0x0F0F0F0F, X.xb.w, dh);
    return L.dm.x * (X.x0 * static_cast<float>(L.sc0 * dl) + X.x1 * static_cast<float>(L.sc1 * dh)) -
           L.dm.y * (X.x0 * static_cast<float>(L.m0 * X.sl) + X.x1 * static_cast<float>(L.m1 * X.sh));
}

__device__ __forceinline__ float q4k_col(const Q4kLane& L, const int8_t* xq, const float* xd) {
    return q4k_dot(L, q4k_x(L, xq, xd));
}

__device__ __forceinline__ float q4k_dp4a_sb(const uint8_t* blk, const int8_t* xq, const float* xd, int t) {
    const Q4kLane L = q4k_lane(blk, t);
    return q4k_col(L, xq, xd);
}

// K-quant rows use 16 lanes (two super-blocks per pass), two rows per warp. With 32 lanes per row a row of
// 10 super-blocks (Qwen3-4B, K = 2560) left a sixth of the lanes idle on its last pass; 2 divides every
// Qwen3 K (2560, 4096, 9728, 12288). Returns this lane's partial sum; the caller reduces over 16 lanes.
template <int LPR>
__device__ __forceinline__ float q4k_dp4a_part(const uint8_t* rowp, const int8_t* xq, const float* xd, int nb, int sub) {
    const int t = sub & 7;
    float acc = 0.f;
#pragma unroll 4
    for (int b = sub >> 3; b < nb; b += LPR / 8) {
        acc += q4k_dp4a_sb(rowp + b * 144, xq + b * 256, xd + b * 8, t);
    }
    return acc;
}

// ---- Q6_K: 210-byte super-block (ql[128], qh[64], scales[16], d). 8 lanes per super-block. ----
struct Q6kLane {
    int q[2][4];  // 6-bit values (0..63) packed four per int, for the two 4-byte words this lane covers
    float sc[4];
    float d;
    int xoff;     // activation byte offset inside the super-block
    int doff;     // activation scale index inside the super-block
};

__device__ __forceinline__ Q6kLane q6k_lane(const uint8_t* blk, int t) {
    Q6kLane L;
    const int h = t >> 2;            // 128-weight half
    const int l0 = (t & 3) << 3;     // 8 consecutive l values
    const uint8_t* ql = blk + h * 64 + l0;
    const uint8_t* qh = blk + 128 + h * 32 + l0;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + h * 8 + (l0 >> 4);
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const int a = qg_ld_b2(ql + 4 * w);
        const int b = qg_ld_b2(ql + 32 + 4 * w);
        const int c = qg_ld_b2(qh + 4 * w);
        L.q[w][0] = (a & 0x0F0F0F0F) | ((c << 4) & 0x30303030);
        L.q[w][1] = (b & 0x0F0F0F0F) | ((c << 2) & 0x30303030);
        L.q[w][2] = ((a >> 4) & 0x0F0F0F0F) | (c & 0x30303030);
        L.q[w][3] = ((b >> 4) & 0x0F0F0F0F) | ((c >> 2) & 0x30303030);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) L.sc[k] = static_cast<float>(sc[2 * k]);
    L.d = qg_half(blk + 208);
    L.xoff = h * 128 + l0;
    L.doff = h * 4;
    return L;
}

// As Q4kX: one lane's activations of one column for one super-block, row-independent.
struct Q6kX {
    int xv[2][4];
    int sum[4];
    float xd[4];
};

__device__ __forceinline__ Q6kX q6k_x(const Q6kLane& L, const int8_t* xq, const float* xd) {
    Q6kX X;
    const int8_t* xb = xq + L.xoff;
    const int ones = 0x01010101;
#pragma unroll
    for (int k = 0; k < 4; ++k) X.sum[k] = 0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            X.xv[w][k] = *reinterpret_cast<const int*>(xb + k * 32 + 4 * w);
            X.sum[k] = __dp4a(ones, X.xv[w][k], X.sum[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) X.xd[k] = xd[L.doff + k];
    return X;
}

__device__ __forceinline__ float q6k_dot(const Q6kLane& L, const Q6kX& X) {
    int dot[4] = {0, 0, 0, 0};
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int k = 0; k < 4; ++k) dot[k] = __dp4a(L.q[w][k], X.xv[w][k], dot[k]);
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc += X.xd[k] * L.sc[k] * static_cast<float>(dot[k] - 32 * X.sum[k]);
    }
    return L.d * acc;
}

__device__ __forceinline__ float q6k_col(const Q6kLane& L, const int8_t* xq, const float* xd) {
    return q6k_dot(L, q6k_x(L, xq, xd));
}

__device__ __forceinline__ float q6k_dp4a_sb(const uint8_t* blk, const int8_t* xq, const float* xd, int t) {
    const Q6kLane L = q6k_lane(blk, t);
    return q6k_col(L, xq, xd);
}

template <int LPR>
__device__ __forceinline__ float q6k_dp4a_part(const uint8_t* rowp, const int8_t* xq, const float* xd, int nb, int sub) {
    const int t = sub & 7;
    float acc = 0.f;
#pragma unroll 4
    for (int b = sub >> 3; b < nb; b += LPR / 8) {
        acc += q6k_dp4a_sb(rowp + b * 210, xq + b * 256, xd + b * 8, t);
    }
    return acc;
}

// ---- Q8_0: 34-byte block (d, 32 x int8). 4 lanes per block, 8 weights per lane. ----
__device__ __forceinline__ float q8_col(int w0, int w1, float d, const int8_t* xb, float xd) {
    int dot = __dp4a(w0, *reinterpret_cast<const int*>(xb), 0);
    dot = __dp4a(w1, *reinterpret_cast<const int*>(xb + 4), dot);
    return d * xd * static_cast<float>(dot);
}

__device__ __forceinline__ float q8_dp4a_row(const uint8_t* rowp, const int8_t* xq, const float* xd, int nb, int lane) {
    const int t = lane & 3;
    float acc = 0.f;
#pragma unroll 4
    for (int b = lane >> 2; b < nb; b += 8) {
        const uint8_t* blk = rowp + b * 34;
        acc += q8_col(qg_ld_b2(blk + 2 + t * 8), qg_ld_b2(blk + 6 + t * 8), qg_half(blk), xq + b * 32 + t * 8, xd[b]);
    }
    return qg_warp_sum(acc);
}

// Rows one warp produces per pass: Q8_0 uses the whole warp per row, the K-quants a half warp.
// lpr: lanes per row for the K-quants, 32 or 16.
__host__ __device__ constexpr int qgemv_rows_per_warp(int type, int lpr = 16) { return type == 0 ? 1 : 32 / lpr; }

// Block-wide row loop. Usable both as a persistent-scheduler task and inside a grid kernel.
// type: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K.
template <int type, int LPR = 16>
__device__ __forceinline__ void qgemv_dp4a_rows(const uint8_t* W, const int8_t* xq, const float* xd, float* y, int rows,
                                                int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nb = type == 0 ? (K >> 5) : (K >> 8);
    const int bb = type == 0 ? 34 : (type == 1 ? 144 : 210);
    const size_t row_bytes = static_cast<size_t>(nb) * bb;
    if (type == 0) {
        for (int row = warp; row < rows; row += nwarps) {
            const float v = q8_dp4a_row(W + static_cast<size_t>(row) * row_bytes, xq, xd, nb, lane);
            if (lane == 0) y[row] = v;
        }
    } else {
        constexpr int RPW = 32 / LPR;
        const int half = lane / LPR;
        const int sub = lane % LPR;
        // Every lane runs the same number of passes so the shuffles below see a full warp.
        for (int base = warp * RPW; base < rows; base += nwarps * RPW) {
            const int row = base + half;
            const bool valid = row < rows;
            const uint8_t* rowp = W + static_cast<size_t>(valid ? row : base) * row_bytes;
            float v = 0.f;
            if (valid) {
                v = type == 1 ? q4k_dp4a_part<LPR>(rowp, xq, xd, nb, sub) : q6k_dp4a_part<LPR>(rowp, xq, xd, nb, sub);
            }
#pragma unroll
            for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (valid && sub == 0) y[row] = v;
        }
    }
    __syncthreads();
}

// NC activation columns against the same rows: the weights are read once. Column c reads xq + c * K,
// xd + c * (K / 32) and writes y[c * ystride + row]. Per column the arithmetic and summation order are the
// single-column path's, so a column's result is bit-identical to qgemv_dp4a_rows on that column alone.
template <int type, int NC>
__device__ __forceinline__ void qgemv_dp4a_rows_mc(const uint8_t* W, const int8_t* xq, const float* xd, float* y,
                                                   int ystride, int rows, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nb = type == 0 ? (K >> 5) : (K >> 8);
    const int bb = type == 0 ? 34 : (type == 1 ? 144 : 210);
    const size_t row_bytes = static_cast<size_t>(nb) * bb;
    const int kd = K >> 5;
    if (type == 0) {
        const int t = lane & 3;
        for (int row = warp; row < rows; row += nwarps) {
            const uint8_t* rowp = W + static_cast<size_t>(row) * row_bytes;
            float acc[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[c] = 0.f;
#pragma unroll 4
            for (int b = lane >> 2; b < nb; b += 8) {
                const uint8_t* blk = rowp + b * 34;
                const int w0 = qg_ld_b2(blk + 2 + t * 8);
                const int w1 = qg_ld_b2(blk + 6 + t * 8);
                const float d = qg_half(blk);
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[c] += q8_col(w0, w1, d, xq + c * K + b * 32 + t * 8, xd[c * kd + b]);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float v = qg_warp_sum(acc[c]);
                if (lane == 0) y[c * ystride + row] = v;
            }
        }
    } else {
        constexpr int LPR = 16;
        const int half = lane / LPR;
        const int sub = lane % LPR;
        const int t = sub & 7;
        for (int base = warp * 2; base < rows; base += nwarps * 2) {
            const int row = base + half;
            const bool valid = row < rows;
            const uint8_t* rowp = W + static_cast<size_t>(valid ? row : base) * row_bytes;
            float acc[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[c] = 0.f;
            if (valid) {
                // One super-block in flight, except 5-7 columns: unrolling there helps now that wide variants get
                // 128 registers (k_gemv_mc launch bounds), but it spills at 8-12 columns. At 64 registers unrolling
                // had pushed Q4_K past 180 registers and cut occupancy (verify time jumped ~60% from T = 5 to 6).
#pragma unroll (NC > 4 && NC < 8 ? 2 : 1)
                for (int b = sub >> 3; b < nb; b += LPR / 8) {
                    if (type == 1) {
                        const Q4kLane L = q4k_lane(rowp + b * 144, t);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[c] += q4k_col(L, xq + c * K + b * 256, xd + c * kd + b * 8);
                    } else {
                        const Q6kLane L = q6k_lane(rowp + b * 210, t);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[c] += q6k_col(L, xq + c * K + b * 256, xd + c * kd + b * 8);
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float v = acc[c];
#pragma unroll
                for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
                if (valid && sub == 0) y[c * ystride + row] = v;
            }
        }
    }
    __syncthreads();
}

// Multi-column passes: as qgemv_dp4a_rows_mc for the K-quants, but each half-warp computes R rows (base + 2 i + half)
// at the same K positions. Activations (and their sums) are loaded once per column
// for both rows, which halves activation traffic per weight; a no-activation-loads experiment had shown loads were
// the largest per-column cost. Each row's arithmetic, accumulation order and 16-lane reduction are those of
// qgemv_dp4a_rows_mc, so results are bit-identical. Rows per warp: 2 R.
// Rows [row0, row_end) of [W; W2] (rows >= rows1 come from W2, chosen per row, so a block may straddle the two).
template <int type, int NC, int R>
__device__ __forceinline__ void qgemv_dp4a_rows_mc2(const uint8_t* W, const uint8_t* W2, int rows1, const int8_t* xq,
                                                    const float* xd, float* y, int ystride, int row0, int row_end,
                                                    int K) {
    static_assert(type == 1 || type == 2, "K-quants only");
    constexpr int LPR = 16;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nb = K >> 8;
    const int bb = type == 1 ? 144 : 210;
    const size_t row_bytes = static_cast<size_t>(nb) * bb;
    const int kd = K >> 5;
    const int half = lane / LPR;
    const int sub = lane % LPR;
    const int t = sub & 7;
    auto row_ptr = [&](int r) {
        return W2 && r >= rows1 ? W2 + static_cast<size_t>(r - rows1) * row_bytes : W + static_cast<size_t>(r) * row_bytes;
    };
    for (int base = row0 + warp * 2 * R; base < row_end; base += nwarps * 2 * R) {
        int rr[R];
        bool vv[R];
        const uint8_t* pr[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            rr[i] = base + 2 * i + half;
            vv[i] = rr[i] < row_end;
            pr[i] = row_ptr(vv[i] ? rr[i] : base);
        }
        float acc[R][NC];
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[i][c] = 0.f;
        if (vv[0]) {
#pragma unroll 1
            for (int b = sub >> 3; b < nb; b += LPR / 8) {
                if (type == 1) {
                    Q4kLane L[R];
#pragma unroll
                    for (int i = 0; i < R; ++i) L[i] = q4k_lane(pr[i] + b * 144, t);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        const Q4kX X = q4k_x(L[0], xq + c * K + b * 256, xd + c * kd + b * 8);
#pragma unroll
                        for (int i = 0; i < R; ++i) acc[i][c] += q4k_dot(L[i], X);
                    }
                } else {
                    Q6kLane L[R];
#pragma unroll
                    for (int i = 0; i < R; ++i) L[i] = q6k_lane(pr[i] + b * 210, t);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        const Q6kX X = q6k_x(L[0], xq + c * K + b * 256, xd + c * kd + b * 8);
#pragma unroll
                        for (int i = 0; i < R; ++i) acc[i][c] += q6k_dot(L[i], X);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                float a = acc[i][c];
#pragma unroll
                for (int o = LPR / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
                if (sub == 0 && vv[i]) y[c * ystride + rr[i]] = a;
            }
        }
    }
    __syncthreads();
}

}  // namespace akd
