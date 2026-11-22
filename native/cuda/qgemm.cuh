#pragma once

// Fused dequantize + fp16 tensor-core GEMM for prefill chunks:
//   Y[c * ldy + r] = sum_k W[r][k] * X[c * K + k]   (r < rows, c < T)
// W is a GGUF Q4_K or Q6_K matrix (rows x K, K % 256 == 0), X fp16 activations [T][K], Y fp32.
//
// The cuBLAS path first expands the whole matrix to fp16 (2 bytes per weight written and read back) for every
// prefill chunk, a fixed ~130-180 ms per chunk for this model regardless of chunk length. Here each block decodes
// the weight tile it needs straight from the 4/6-bit blocks into shared memory (same rounding as pf_dequant_f16),
// so no fp16 copy of the matrix exists.
//
// Block: 128 weight rows x BN tokens (64 or 128), K in tiles of 64 (one Q4_K 64-weight group / half a Q6_K
// 128-weight half). 256 threads = 8 warps in a 4 (rows) x 2 (tokens) grid, each 32 rows x BN/2 tokens via
// mma.m16n8k16 (fp16 in, fp32 accumulate). Two threads decode each weight row of a tile (16 bytes of nibbles each);
// every weight is decoded once per BN tokens. The next tile's raw bytes and activations are fetched into
// registers while the current tile is multiplied.
//
// RTX 4060, this model, GEMM ms per prefill chunk (cuBLAS incl. fp16 expansion -> BN 64 / BN 128):
//   T = 64: 189 -> 63 / 103   T = 128: 198 -> 101 / 105   T = 256: 244 -> 186 / 175   T = 512: 354 -> 353 / 320
//   T = 1024: 572 -> 703 / 611   T = 2048: 1018 -> 1390 / 1213
// so pf_gemm uses BN 64 up to 128 columns, BN 128 up to 768, cuBLAS beyond.

#include <cuda_fp16.h>
#include <cstdint>

#include "qgemv.cuh"

namespace akd {
namespace qgemm {

constexpr int BM = 128;
constexpr int BK = 64;
constexpr int LD = BK + 8;  // padded smem row (halves): 36 words, conflict-free fragment loads
constexpr int NT = 256;

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t ld32(const __half* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ int byte_of(uint32_t w, int i) { return static_cast<int>((w >> (8 * i)) & 0xFFu); }

// Raw bytes one thread needs for its half of a weight row in one K tile.
struct Raw {
    uint32_t r[13];
};

// Q4_K scale byte i (0..11) from r[4..6], selected with constant indices: indexing the array with a runtime value
// would move it to local memory.
__device__ __forceinline__ int scale_byte(const Raw& R, int i) {
    const uint32_t w = i < 4 ? R.r[4] : (i < 8 ? R.r[5] : R.r[6]);
    return static_cast<int>((w >> (8 * (i & 3))) & 0xFFu);
}

template <int type>
__device__ __forceinline__ void load_raw(Raw& R, const uint8_t* W, int rows, int K, int row, int half, int k0) {
    if (row >= rows) {
#pragma unroll
        for (int i = 0; i < 13; ++i) R.r[i] = 0u;
        return;
    }
    const int b = k0 >> 8;           // super-block
    const int sub = (k0 & 255) >> 6;  // 64-wide tile inside it
    if (type == 1) {                 // Q4_K: d, dmin | scales[12] | qs[128]
        const uint8_t* blk = W + static_cast<size_t>(row) * (K / 256) * 144 + b * 144;
        const uint4 q = *reinterpret_cast<const uint4*>(blk + 16 + (2 * sub + half) * 16);
        R.r[0] = q.x;
        R.r[1] = q.y;
        R.r[2] = q.z;
        R.r[3] = q.w;
        const uint32_t* w32 = reinterpret_cast<const uint32_t*>(blk);
        R.r[4] = w32[1];
        R.r[5] = w32[2];
        R.r[6] = w32[3];
        R.r[7] = w32[0];
    } else {  // Q6_K: ql[128] | qh[64] | scales[16] | d
        const uint8_t* blk = W + static_cast<size_t>(row) * (K / 256) * 210 + b * 210;
        const int h = sub >> 1;
        const int odd = sub & 1;
        const int l0 = half * 16;
        const uint8_t* ql = blk + h * 64 + l0;
        const uint8_t* qh = blk + 128 + h * 32 + l0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            R.r[w] = static_cast<uint32_t>(qg_ld_b2(ql + 4 * w));
            R.r[4 + w] = static_cast<uint32_t>(qg_ld_b2(ql + 32 + 4 * w));
            R.r[8 + w] = static_cast<uint32_t>(qg_ld_b2(qh + 4 * w));
        }
        const uint8_t* sc = blk + 192 + h * 8;
        const uint32_t sa = sc[(odd ? 4 : 0) + half];
        const uint32_t sb = sc[(odd ? 6 : 2) + half];
        const uint32_t dbits = *reinterpret_cast<const uint16_t*>(blk + 208);
        R.r[12] = sa | (sb << 8) | (dbits << 16);
    }
}

// Decode the raw bytes into 32 fp16 weights at Ws[row][...] (k positions off+i and 32+off+i, i < 16).
template <int type>
__device__ __forceinline__ void store_w(const Raw& R, __half* Ws, int row, int half, int k0) {
    float lo[16], hi[16];
    const int off = half * 16;
    if (type == 1) {
        const int j = (k0 & 255) >> 6;
        const int is = 2 * j;
        auto s = [&](int i) { return scale_byte(R, i); };
        int sc0, m0, sc1, m1;
        if (is < 4) {
            sc0 = s(is) & 63;
            m0 = s(is + 4) & 63;
            sc1 = s(is + 1) & 63;
            m1 = s(is + 5) & 63;
        } else {
            sc0 = (s(is + 4) & 0xF) | ((s(is - 4) >> 6) << 4);
            m0 = (s(is + 4) >> 4) | ((s(is) >> 6) << 4);
            sc1 = (s(is + 5) & 0xF) | ((s(is - 3) >> 6) << 4);
            m1 = (s(is + 5) >> 4) | ((s(is + 1) >> 6) << 4);
        }
        const uint32_t dm = R.r[7];
        __half2 dmh = *reinterpret_cast<const __half2*>(&dm);
        const float2 f = __half22float2(dmh);
        const float a0 = f.x * static_cast<float>(sc0), b0 = f.y * static_cast<float>(m0);
        const float a1 = f.x * static_cast<float>(sc1), b1 = f.y * static_cast<float>(m1);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = byte_of(R.r[i >> 2], i & 3);
            lo[i] = a0 * static_cast<float>(q & 0xF) - b0;
            hi[i] = a1 * static_cast<float>(q >> 4) - b1;
        }
    } else {
        const int odd = ((k0 & 255) >> 6) & 1;
        const int8_t sa = static_cast<int8_t>(R.r[12] & 0xFFu);
        const int8_t sb = static_cast<int8_t>((R.r[12] >> 8) & 0xFFu);
        const uint16_t dbits = static_cast<uint16_t>(R.r[12] >> 16);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dbits));
        const float da = d * static_cast<float>(sa), db = d * static_cast<float>(sb);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int a = byte_of(R.r[i >> 2], i & 3);
            const int bq = byte_of(R.r[4 + (i >> 2)], i & 3);
            const int c = byte_of(R.r[8 + (i >> 2)], i & 3);
            int q0, q1;
            if (!odd) {
                q0 = (a & 0xF) | ((c & 3) << 4);
                q1 = (bq & 0xF) | (((c >> 2) & 3) << 4);
            } else {
                q0 = (a >> 4) | (((c >> 4) & 3) << 4);
                q1 = (bq >> 4) | (((c >> 6) & 3) << 4);
            }
            lo[i] = da * static_cast<float>(q0 - 32);
            hi[i] = db * static_cast<float>(q1 - 32);
        }
    }
    uint4* dst_lo = reinterpret_cast<uint4*>(Ws + row * LD + off);
    uint4* dst_hi = reinterpret_cast<uint4*>(Ws + row * LD + 32 + off);
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        dst_lo[v] = make_uint4(pack2(lo[8 * v], lo[8 * v + 1]), pack2(lo[8 * v + 2], lo[8 * v + 3]),
                               pack2(lo[8 * v + 4], lo[8 * v + 5]), pack2(lo[8 * v + 6], lo[8 * v + 7]));
        dst_hi[v] = make_uint4(pack2(hi[8 * v], hi[8 * v + 1]), pack2(hi[8 * v + 2], hi[8 * v + 3]),
                               pack2(hi[8 * v + 4], hi[8 * v + 5]), pack2(hi[8 * v + 6], hi[8 * v + 7]));
    }
}

// grid (ceil(T / BN), ceil(rows / BM)): token tiles of the same weight rows run side by side and share L2.
template <int type, int BN>
__global__ void __launch_bounds__(NT, 2) k_qgemm(const uint8_t* W, const __half* X, float* Y, int rows, int K, int T,
                                              int ldy) {
    static_assert(BN == 64 || BN == 128, "token tile");
    __shared__ __align__(16) __half Ws[BM * LD];
    __shared__ __align__(16) __half Xs[BN * LD];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gq = lane >> 2;
    const int tig = lane & 3;
    const int c0 = blockIdx.x * BN;
    const int r0 = blockIdx.y * BM;
    const int wrow = tid >> 1;  // weight row this thread decodes
    const int whalf = tid & 1;
    const int wm = warp >> 1;   // 32-row slab of the warp
    const int wn = warp & 1;    // BN / 2-token slab

    Raw R;
    uint4 xr[BN / 32];
    auto load_x = [&](int k0) {
#pragma unroll
        for (int i = 0; i < BN / 32; ++i) {
            const int idx = tid + i * NT;
            const int n = idx >> 3;
            const int kk = (idx & 7) * 8;
            const int c = c0 + n;
            xr[i] = c < T ? *reinterpret_cast<const uint4*>(X + static_cast<size_t>(c) * K + k0 + kk)
                          : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto store_x = [&]() {
#pragma unroll
        for (int i = 0; i < BN / 32; ++i) {
            const int idx = tid + i * NT;
            *reinterpret_cast<uint4*>(Xs + (idx >> 3) * LD + (idx & 7) * 8) = xr[i];
        }
    };

    float acc[2][BN / 16][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < BN / 16; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

    const int nk = K / BK;
    load_raw<type>(R, W, rows, K, r0 + wrow, whalf, 0);
    load_x(0);
    store_w<type>(R, Ws, wrow, whalf, 0);
    store_x();
    __syncthreads();
    for (int kt = 0; kt < nk; ++kt) {
        const int k_next = (kt + 1) * BK;
        if (kt + 1 < nk) {  // fetch the next tile while this one is multiplied
            load_raw<type>(R, W, rows, K, r0 + wrow, whalf, k_next);
            load_x(k_next);
        }
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t a[2][4], b[BN / 16][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const __half* p = Ws + (wm * 32 + mt * 16 + gq) * LD + ks * 16 + tig * 2;
                a[mt][0] = ld32(p);
                a[mt][1] = ld32(p + 8 * LD);
                a[mt][2] = ld32(p + 8);
                a[mt][3] = ld32(p + 8 * LD + 8);
            }
#pragma unroll
            for (int nt = 0; nt < BN / 16; ++nt) {
                const __half* p = Xs + (wn * (BN / 2) + nt * 8 + gq) * LD + ks * 16 + tig * 2;
                b[nt][0] = ld32(p);
                b[nt][1] = ld32(p + 8);
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < BN / 16; ++nt) mma16816(acc[mt][nt], a[mt], b[nt]);
        }
        __syncthreads();
        if (kt + 1 < nk) {
            store_w<type>(R, Ws, wrow, whalf, k_next);
            store_x();
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < BN / 16; ++nt) {
            const int r = r0 + wm * 32 + mt * 16 + gq;
            const int c = c0 + wn * (BN / 2) + nt * 8 + tig * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = r + (e >> 1) * 8;
                const int cc = c + (e & 1);
                if (rr < rows && cc < T) Y[static_cast<size_t>(cc) * ldy + rr] = acc[mt][nt][e];
            }
        }
    }
}

// ================================================================================================
// int8 tensor-core variant. Activations are quantized per 32 values (xq int8, scale xd, and xsd = xd * sum(xq) for the
// Q4_K min term) - the decode path's Q8_1 arithmetic - and weights are decoded to integers in shared memory:
//   Q4_K: nibbles 0..15 straight from the bytes (mask), one mma.m16n8k32.s8 per 32-wide sub-block, then
//         acc += d*sc * xd * dot - dmin*m * xsd
//   Q6_K: q - 32 (per-byte subtract), one mma.m16n8k16.s8 per 16-wide scale group, acc += d*sc * xd * dot
// int8 mma runs at several times the rate of fp16 with fp32 accumulation on Ada GeForce parts, but measured here it
// barely helps: per 32 values every output element needs int -> float conversion plus the weight / activation scales
// (and the min term), and that scalar work, not the tensor cores, bounds the kernel. RTX 4060 GEMM ms per chunk,
// fp16 fused -> int8, BN 64: T = 32: 58.5 -> 49.6, 64: 60.1 -> 52.8, 128: 98.6 -> 86.7. With BN 128 the int8 kernel
// needs ~186 registers (1 block / SM) or spills at 128, and is slower than fp16 (T = 512: 310 vs 457 ms). It is an
// opt-in for chunks of <= 128 tokens (AK_Q35_INT8=1); accuracy is the decode path's (logit correlation 0.9994 vs
// 0.99999 for the fp16 kernels). Getting llama.cpp-class int8 throughput needs integer accumulation across scale
// blocks or a deeper cp.async pipeline.
// ================================================================================================

constexpr int LDQ = BK + 16;  // int8 smem row: 80 bytes = 20 words, conflict-free 32-bit fragment loads

__device__ __forceinline__ void mma_s8_k32(int* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_s8_k16(int* c, const uint32_t* a, uint32_t b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ uint32_t ldq32(const int8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// int -> float of an int8 mma sum (exact either way: |v| < 65536). AK_MMQ_I2F = 1 (default): one I2F. 0: the two
// full-rate ops of 1.5 * 2^23 + v as float bits, minus 1.5 * 2^23. The 2-op form was chosen because I2F runs at a
// quarter rate, but Nsight Compute showed these kernels are bound by instruction issue (2 warps per scheduler, issue
// slots ~55% busy) with the conversion pipe mostly idle: one instruction instead of two made Q6_K 12% faster and
// Q4_K 1-2% (docs/QWEN38_9B.md, "Profiling the int8 GEMM").
#ifndef AK_MMQ_I2F
#define AK_MMQ_I2F 1
#endif
__device__ __forceinline__ float i2f_small(int v) {
    if constexpr (AK_MMQ_I2F) return __int2float_rn(v);
    return __int_as_float(v + 0x4B400000) - 12582912.0f;
}

// fp16 activations [T][K] -> int8 [T][K], xd / xsd [T][K / 32]. grid (K / 256, T), 256 threads (a warp per block).
__global__ void __launch_bounds__(256) k_quant_x8(const __half* X, int K, int8_t* xq, float* xd, float* xsd) {
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * 8 + (threadIdx.x >> 5);
    const size_t c = blockIdx.y;
    const size_t i = c * K + static_cast<size_t>(blk) * 32 + lane;
    const float x = __half2float(X[i]);
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = amax / 127.f;
    const int q = d > 0.f ? __float2int_rn(x / d) : 0;
    int s = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    xq[i] = static_cast<int8_t>(q);
    if (lane == 0) {
        const size_t j = c * (K / 32) + blk;
        xd[j] = d;
        xsd[j] = d * static_cast<float>(s);
    }
}

// One thread fetches 1/P of a weight row's 32-byte slice of a 64-wide K tile (P = 2: 16 bytes, P = 4: 8 bytes),
// plus what it needs to write the row's scales.
template <int type, int P>
__device__ __forceinline__ void load_raw8(Raw& R, const uint8_t* W, int rows, int K, int row, int part, int k0) {
    constexpr int WPP = 8 / P;  // 32-bit words of each byte stream per part
    if (row >= rows) {
#pragma unroll
        for (int i = 0; i < 13; ++i) R.r[i] = 0u;
        return;
    }
    const int b = k0 >> 8;
    const int sub = (k0 & 255) >> 6;
    if (type == 1) {
        const uint8_t* blk = W + static_cast<size_t>(row) * (K / 256) * 144 + b * 144;
        const uint8_t* q = blk + 16 + sub * 32 + part * (32 / P);
        if (WPP == 4) {
            const uint4 v = *reinterpret_cast<const uint4*>(q);
            R.r[0] = v.x;
            R.r[1] = v.y;
            R.r[2] = v.z;
            R.r[3] = v.w;
        } else {
            const uint2 v = *reinterpret_cast<const uint2*>(q);
            R.r[0] = v.x;
            R.r[1] = v.y;
        }
        if (part == 0) {
            const uint4 h = *reinterpret_cast<const uint4*>(blk);
            R.r[7] = h.x;
            R.r[4] = h.y;
            R.r[5] = h.z;
            R.r[6] = h.w;
        }
    } else {
        const uint8_t* blk = W + static_cast<size_t>(row) * (K / 256) * 210 + b * 210;
        const int h = sub >> 1;
        const int odd = sub & 1;
        const int l0 = part * (32 / P);
        const uint8_t* ql = blk + h * 64 + l0;
        const uint8_t* qh = blk + 128 + h * 32 + l0;
#pragma unroll
        for (int w = 0; w < WPP; ++w) {
            R.r[w] = static_cast<uint32_t>(qg_ld_b2(ql + 4 * w));
            R.r[4 + w] = static_cast<uint32_t>(qg_ld_b2(ql + 32 + 4 * w));
            R.r[8 + w] = static_cast<uint32_t>(qg_ld_b2(qh + 4 * w));
        }
        if ((l0 & 15) == 0) {
            const int g = l0 >> 4;
            const uint8_t* sc = blk + 192 + h * 8;
            const uint32_t sa = sc[(odd ? 4 : 0) + g];
            const uint32_t sb = sc[(odd ? 6 : 2) + g];
            const uint32_t dbits = *reinterpret_cast<const uint16_t*>(blk + 208);
            R.r[12] = sa | (sb << 8) | (dbits << 16);
        }
    }
}

// Integer decode into Wq[row][...] (k = off.. and 32 + off..) and the row's per-block float scales into wsc.
template <int type, int P>
__device__ __forceinline__ void store_w8(const Raw& R, int8_t* Wq, float* wsc, int row, int part, int k0) {
    constexpr int WPP = 8 / P;
    const int off = part * (32 / P);
    uint32_t lo[4], hi[4];
    if (type == 1) {
#pragma unroll
        for (int w = 0; w < WPP; ++w) {
            lo[w] = R.r[w] & 0x0F0F0F0Fu;
            hi[w] = (R.r[w] >> 4) & 0x0F0F0F0Fu;
        }
        if (part == 0) {
            const int is = 2 * ((k0 & 255) >> 6);
            auto s = [&](int i) { return scale_byte(R, i); };
            int sc0, m0, sc1, m1;
            if (is < 4) {
                sc0 = s(is) & 63;
                m0 = s(is + 4) & 63;
                sc1 = s(is + 1) & 63;
                m1 = s(is + 5) & 63;
            } else {
                sc0 = (s(is + 4) & 0xF) | ((s(is - 4) >> 6) << 4);
                m0 = (s(is + 4) >> 4) | ((s(is) >> 6) << 4);
                sc1 = (s(is + 5) & 0xF) | ((s(is - 3) >> 6) << 4);
                m1 = (s(is + 5) >> 4) | ((s(is + 1) >> 6) << 4);
            }
            const uint32_t dm = R.r[7];
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&dm));
            *reinterpret_cast<float4*>(wsc + row * 4) =
                make_float4(f.x * static_cast<float>(sc0), f.x * static_cast<float>(sc1), f.y * static_cast<float>(m0),
                            f.y * static_cast<float>(m1));
        }
    } else {
        const int odd = ((k0 & 255) >> 6) & 1;
#pragma unroll
        for (int w = 0; w < WPP; ++w) {
            const uint32_t a = R.r[w], b = R.r[4 + w], c = R.r[8 + w];
            const uint32_t v0 = odd ? (((a >> 4) & 0x0F0F0F0Fu) | (c & 0x30303030u))
                                    : ((a & 0x0F0F0F0Fu) | ((c << 4) & 0x30303030u));
            const uint32_t v1 = odd ? (((b >> 4) & 0x0F0F0F0Fu) | ((c >> 2) & 0x30303030u))
                                    : ((b & 0x0F0F0F0Fu) | ((c << 2) & 0x30303030u));
            lo[w] = __vsub4(v0, 0x20202020u);
            hi[w] = __vsub4(v1, 0x20202020u);
        }
        if ((off & 15) == 0) {
            const int g = off >> 4;  // 16-group of the first quarter; +2: of the second
            const int8_t sa = static_cast<int8_t>(R.r[12] & 0xFFu);
            const int8_t sb = static_cast<int8_t>((R.r[12] >> 8) & 0xFFu);
            const uint16_t dbits = static_cast<uint16_t>(R.r[12] >> 16);
            const float d = __half2float(*reinterpret_cast<const __half*>(&dbits));
            wsc[row * 4 + g] = d * static_cast<float>(sa);
            wsc[row * 4 + 2 + g] = d * static_cast<float>(sb);
        }
    }
    if (WPP == 4) {
        *reinterpret_cast<uint4*>(Wq + row * LDQ + off) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
        *reinterpret_cast<uint4*>(Wq + row * LDQ + 32 + off) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    } else {
        *reinterpret_cast<uint2*>(Wq + row * LDQ + off) = make_uint2(lo[0], lo[1]);
        *reinterpret_cast<uint2*>(Wq + row * LDQ + 32 + off) = make_uint2(hi[0], hi[1]);
    }
}

// Block BM8 weight rows x BN8 tokens, 8 warps in a (BM8 / 32) x (8 / (BM8 / 32)) grid, each warp 32 rows x TW
// tokens. grid (ceil(T / BN8), ceil(rows / BM8)). xq / xd / xsd from k_quant_x8 on the same K.
//   <128, 64>: warps 32 x 32 (short chunks)   <64, 128>: warps 32 x 32 (long chunks, 32 accumulators per thread)
template <int type, int BM8, int BN8>
__global__ void __launch_bounds__(NT, (BN8 * BM8) <= 8192 ? 2 : 1)
    k_qgemm8(const uint8_t* W, const int8_t* xq, const float* xd, const float* xsd, float* Y, int rows, int K, int T,
             int ldy, int diag) {
    constexpr int P = NT / BM8;          // threads decoding each weight row
    constexpr int WM = BM8 / 32;         // warp rows
    constexpr int WN = 8 / WM;           // warp columns
    constexpr int TW = BN8 / WN;         // tokens per warp
    constexpr int NTW = TW / 8;          // 8-token mma tiles per warp
    static_assert(P == 2 || P == 4, "rows per block");
    static_assert(TW % 8 == 0 && NTW >= 1, "token tile");
    __shared__ __align__(16) int8_t Wq[BM8 * LDQ];
    __shared__ __align__(16) int8_t Xq[BN8 * LDQ];
    __shared__ __align__(16) float wsc[BM8 * 4];  // Q4_K: a0 a1 b0 b1; Q6_K: s of the four 16-groups
    __shared__ __align__(16) float xsc[BN8 * 4];  // xd0 xd1 xsd0 xsd1
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gq = lane >> 2;
    const int tig = lane & 3;
    const int c0 = blockIdx.x * BN8;
    const int r0 = blockIdx.y * BM8;
    const int wrow = tid / P;
    const int wpart = tid % P;
    const int wm = warp / WN;
    const int wn = warp % WN;
    const int KB = K / 32;

    Raw R;
    constexpr int XV = (BN8 * 4 + NT - 1) / NT;  // uint4 of int8 activations per thread per tile
    uint4 xr[XV];
    float4 xs4 = make_float4(0.f, 0.f, 0.f, 0.f);
    auto load_x = [&](int k0) {
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int idx = tid + i * NT;
            const int n = idx >> 2;
            const int c = c0 + n;
            xr[i] = (n < BN8 && c < T)
                        ? *reinterpret_cast<const uint4*>(xq + static_cast<size_t>(c) * K + k0 + (idx & 3) * 16)
                        : make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid < BN8) {
            const int c = c0 + tid;
            if (c < T) {
                const float2 a = *reinterpret_cast<const float2*>(xd + static_cast<size_t>(c) * KB + k0 / 32);
                const float2 b = *reinterpret_cast<const float2*>(xsd + static_cast<size_t>(c) * KB + k0 / 32);
                xs4 = make_float4(a.x, a.y, b.x, b.y);
            } else {
                xs4 = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    };
    auto store_x = [&]() {
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int idx = tid + i * NT;
            if ((idx >> 2) < BN8) *reinterpret_cast<uint4*>(Xq + (idx >> 2) * LDQ + (idx & 3) * 16) = xr[i];
        }
        if (tid < BN8) *reinterpret_cast<float4*>(xsc + tid * 4) = xs4;
    };

    float acc[2][NTW][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NTW; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

    const int nk = K / BK;
    load_raw8<type, P>(R, W, rows, K, r0 + wrow, wpart, 0);
    load_x(0);
    store_w8<type, P>(R, Wq, wsc, wrow, wpart, 0);
    store_x();
    __syncthreads();
    for (int kt = 0; kt < nk; ++kt) {
        const int k_next = (kt + 1) * BK;
        if (kt + 1 < nk) {
            load_raw8<type, P>(R, W, rows, K, r0 + wrow, wpart, k_next);
            load_x(k_next);
        }
        if (diag == 1) {
            // timing diagnostic: loads, decode and syncs only
        } else if (type == 1) {
#pragma unroll
            for (int sb = 0; sb < 2; ++sb) {
                uint32_t a[2][4];
                float ra[2][2], rb[2][2];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int r = wm * 32 + mt * 16 + gq;
                    const int8_t* p = Wq + r * LDQ + sb * 32 + tig * 4;
                    a[mt][0] = ldq32(p);
                    a[mt][1] = ldq32(p + 8 * LDQ);
                    a[mt][2] = ldq32(p + 16);
                    a[mt][3] = ldq32(p + 8 * LDQ + 16);
                    ra[mt][0] = wsc[r * 4 + sb];
                    ra[mt][1] = wsc[(r + 8) * 4 + sb];
                    rb[mt][0] = wsc[r * 4 + 2 + sb];
                    rb[mt][1] = wsc[(r + 8) * 4 + 2 + sb];
                }
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt) {
                    const int n = wn * TW + nt * 8;
                    const int8_t* p = Xq + (n + gq) * LDQ + sb * 32 + tig * 4;
                    const uint32_t b[2] = {ldq32(p), ldq32(p + 16)};
                    const float xd0 = xsc[(n + tig * 2) * 4 + sb], xd1 = xsc[(n + tig * 2 + 1) * 4 + sb];
                    const float xs0 = xsc[(n + tig * 2) * 4 + 2 + sb], xs1 = xsc[(n + tig * 2 + 1) * 4 + 2 + sb];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c[4] = {0, 0, 0, 0};
                        mma_s8_k32(c, a[mt], b);
                        float* o = acc[mt][nt];
                        o[0] = fmaf(ra[mt][0] * xd0, i2f_small(c[0]), fmaf(-rb[mt][0], xs0, o[0]));
                        o[1] = fmaf(ra[mt][0] * xd1, i2f_small(c[1]), fmaf(-rb[mt][0], xs1, o[1]));
                        o[2] = fmaf(ra[mt][1] * xd0, i2f_small(c[2]), fmaf(-rb[mt][1], xs0, o[2]));
                        o[3] = fmaf(ra[mt][1] * xd1, i2f_small(c[3]), fmaf(-rb[mt][1], xs1, o[3]));
                    }
                }
            }
        } else {
#pragma unroll
            for (int g16 = 0; g16 < 4; ++g16) {
                uint32_t a[2][2];
                float rs[2][2];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int r = wm * 32 + mt * 16 + gq;
                    const int8_t* p = Wq + r * LDQ + g16 * 16 + tig * 4;
                    a[mt][0] = ldq32(p);
                    a[mt][1] = ldq32(p + 8 * LDQ);
                    rs[mt][0] = wsc[r * 4 + g16];
                    rs[mt][1] = wsc[(r + 8) * 4 + g16];
                }
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt) {
                    const int n = wn * TW + nt * 8;
                    const uint32_t b = ldq32(Xq + (n + gq) * LDQ + g16 * 16 + tig * 4);
                    const float xd0 = xsc[(n + tig * 2) * 4 + (g16 >> 1)];
                    const float xd1 = xsc[(n + tig * 2 + 1) * 4 + (g16 >> 1)];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c[4] = {0, 0, 0, 0};
                        mma_s8_k16(c, a[mt], b);
                        float* o = acc[mt][nt];
                        o[0] = fmaf(rs[mt][0] * xd0, i2f_small(c[0]), o[0]);
                        o[1] = fmaf(rs[mt][0] * xd1, i2f_small(c[1]), o[1]);
                        o[2] = fmaf(rs[mt][1] * xd0, i2f_small(c[2]), o[2]);
                        o[3] = fmaf(rs[mt][1] * xd1, i2f_small(c[3]), o[3]);
                    }
                }
            }
        }
        __syncthreads();
        if (kt + 1 < nk) {
            store_w8<type, P>(R, Wq, wsc, wrow, wpart, k_next);
            store_x();
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            const int r = r0 + wm * 32 + mt * 16 + gq;
            const int c = c0 + wn * TW + nt * 8 + tig * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = r + (e >> 1) * 8;
                const int cc = c + (e & 1);
                if (rr < rows && cc < T) Y[static_cast<size_t>(cc) * ldy + rr] = acc[mt][nt][e];
            }
        }
    }
}

// ================================================================================================
// Pipelined Q4_K int8 GEMM. The plain kernels above stall: loading + decoding + syncing alone took 143 of 276 ms per
// 512-token chunk, at ~75 GB/s of weight traffic, and none of it overlapped the math. Here:
//   - cp.async copies raw Q4_K bytes (16-byte header + the 64 nibble bytes of a 128-wide K half), int8 activations
//     and their scales straight into a STAGES-deep shared-memory ring, so later tiles are in flight while one is
//     multiplied;
//   - there is no decode pass: an mma A operand is a 4-byte nibble word masked (& 0x0F0F0F0F) or shifted (>> 4);
//   - per stage the 128 x 4 sub-block scales are decoded once, cooperatively, into shared memory;
//   - stages are 128 wide in K, so a block syncs twice per 128 K instead of twice per 64.
// Block 128 rows x BN tokens, 8 warps in 4 (rows) x 2 (tokens). Requires K % 256 == 0.
// ================================================================================================

constexpr int P4_BK = 128;          // K per stage
constexpr int P4_WS = 80;           // smem bytes per weight row and stage: header 16 + nibbles 64
constexpr int P4_XS = P4_BK + 16;   // smem bytes per token row and stage (padded)
// stages: 2 for 128-token tiles (3 would exceed 99 KB of shared memory), 4 for 64-token tiles

template <int BN>
__host__ __device__ constexpr size_t p4_stage_bytes() {
    return static_cast<size_t>(BM) * P4_WS + static_cast<size_t>(BN) * P4_XS + static_cast<size_t>(BN) * 8 * 4;
}
// Row stride (floats) of the decoded-scale table of k_mmq4 / k_mmq6: 8 values per row padded to 12, so the 8 rows a
// warp reads with one float4 load each (lanes gq = 0..7) land on 8 different bank groups; at 8 floats (32 bytes) rows
// gq and gq + 4 shared banks (a 2-way conflict on every scale load, Nsight Compute).
constexpr int WSC_LD = 12;

template <int BN, int S, int WN = 2>
__host__ __device__ constexpr size_t p4_smem_bytes() {
    return S * p4_stage_bytes<BN>() + static_cast<size_t>(BM) * WSC_LD * 4;  // + decoded scales
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// DIAG (benchmarks/gemm_bench.cu only; the runtime uses 0), all timing only with wrong results:
//   1 = the per-sub-block float epilogue reduced to one add per output, 2 = no min term,
//   4 = no mma either (loads, syncs and fragment reads only), 5 = no global -> shared copies (compute only),
//   6 = no per-stage scale decode and no barrier after it.
// GLU (FFN gate + up + SiLU in one launch): W is the gate matrix, Wu the up matrix (same shape), rows their row
// count. Each warp's 32-row slab is 16 gate rows and the same 16 up rows, so m-tile 0 holds gate and m-tile 1 up for
// the same outputs in the same thread, and the epilogue writes fp16 silu(gate) * up to Y (as __half, ldy apart)
// instead of two fp32 results that k_silu_quant would read back. A block covers BM / 2 = 64 rows. The accumulation is
// k_mmq4's, and silu * up is k_silu_quant's expression, so the fp16 output is bit-identical to the two-step path.
template <int BN, int S, int MB = 1, int DIAG = 0, bool GLU = false, int WN = 2>
__global__ void __launch_bounds__(128 * WN, MB) k_mmq4(const uint8_t* W, const int8_t* xq, const float* xd,
                                               const float* xsd, float* Y, int rows, int K, int T, int ldy,
                                               int nsplit, const uint8_t* Wu = nullptr) {
    // WN warps along tokens (4 rows of warps x WN): 2 = 8 warps of 32 x BN/2, 4 = 16 warps of 32 x BN/4. More warps
    // per scheduler hide the IMMA-pair and shared-load latencies that left issue slots 45% idle with 8 (Nsight
    // Compute, docs/QWEN38_9B.md "Profiling the int8 GEMM"); each warp then needs half the accumulators.
    constexpr int NTH = 128 * WN;
    constexpr int TW = BN / WN;  // tokens per warp
    constexpr int NTW = TW / 8;
    constexpr size_t SB = p4_stage_bytes<BN>();
    extern __shared__ __align__(16) uint8_t p4_smem[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    // decoded sub-block scales, [row][8]: a of 4 sub-blocks | b of 4. WN = 2: one table for the block's BM rows;
    // WN = 4: each warp's private slab for its own 32 rows (see the decode below).
    float* wsc = reinterpret_cast<float*>(p4_smem + S * SB);
    const int lane = tid & 31;
    const int gq = lane >> 2;
    const int tig = lane & 3;
    const int wm = warp / WN;
    const int wn = warp % WN;
    const int c0 = blockIdx.x * BN;
    const int r0 = blockIdx.y * (GLU ? BM / 2 : BM);
    const int KB = K / 32;
    const size_t row_bytes = static_cast<size_t>(K / 256) * 144;
    // Tile row tr -> matrix row: plain, tr; GLU, slab tr / 32 holds 16 gate rows then the same 16 up rows.
    auto src_row = [&](int tr, const uint8_t*& mat) {
        if constexpr (GLU) {
            mat = (tr & 16) ? Wu : W;
            return (tr >> 5) * 16 + (tr & 15);
        } else {
            mat = W;
            return tr;
        }
    };
    // Split-K: block z covers stages [kt0, kt0 + nk) and writes its partial sums to plane z of Y (k_splitk_sum
    // adds the planes in a fixed order, so results are deterministic).
    const int nk = K / P4_BK / nsplit;
    const int kt0 = static_cast<int>(blockIdx.z) * nk;
    Y += static_cast<size_t>(blockIdx.z) * T * ldy;

    auto wraw = [&](int st) { return p4_smem + st * SB; };
    auto xsm = [&](int st) { return reinterpret_cast<int8_t*>(p4_smem + st * SB + BM * P4_WS); };
    auto xsc = [&](int st) { return reinterpret_cast<float*>(p4_smem + st * SB + BM * P4_WS + BN * P4_XS); };

    auto issue = [&](int kt) {
        if constexpr (DIAG == 5) return;
        const int st = kt % S;
        const int k0 = kt * P4_BK;
        const int b = k0 >> 8;
        const int h = (k0 >> 7) & 1;
        uint8_t* ws = wraw(st);
        for (int i = tid; i < BM * 5; i += NTH) {
            const int r = i / 5, ch = i - r * 5;
            const uint8_t* mat;
            const int mr = r0 + src_row(r, mat);
            const bool ok = mr < rows;
            const uint8_t* blk = mat + (ok ? mr * row_bytes : 0) + b * 144;
            cp_async16(ws + r * P4_WS + ch * 16, ch == 0 ? blk : blk + 16 + h * 64 + (ch - 1) * 16, ok);
        }
        int8_t* xs = xsm(st);
        for (int i = tid; i < BN * (P4_BK / 16); i += NTH) {
            const int n = i >> 3, ch = i & 7;
            const bool ok = c0 + n < T;
            cp_async16(xs + n * P4_XS + ch * 16, xq + (ok ? static_cast<size_t>(c0 + n) * K : 0) + k0 + ch * 16, ok);
        }
        float* sc = xsc(st);  // per token: [xd of the stage's four 32-blocks | xsd of them]
        for (int i = tid; i < BN * 2; i += NTH) {
            const int n = i >> 1, which = i & 1;
            const bool ok = c0 + n < T;
            const float* src = (which ? xsd : xd) + (ok ? static_cast<size_t>(c0 + n) * KB : 0) + k0 / 32;
            cp_async16(sc + n * 8 + which * 4, src, ok);
        }
    };

    float acc[2][NTW][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NTW; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < nk) issue(kt0 + s);
        cp_async_commit();
    }
    for (int kt = kt0; kt < kt0 + nk; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();  // stage kt landed for every thread; stage kt - 1 and wsc are free
        const int st = kt % S;
        const int h = ((kt * P4_BK) >> 7) & 1;
        // Sub-block scales of this stage (global sub-block 4h + sb), decoded by the whole block, then a barrier.
        if constexpr (DIAG != 6) {
            const uint8_t* ws = wraw(st);
            for (int i = tid; i < BM * 4; i += NTH) {
                const int r = i >> 2, sb = i & 3;
                const uint8_t* hd = ws + r * P4_WS;
                const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(hd));
                const uint8_t* q = hd + 4;
                const int j = 4 * h + sb;
                int sc, mn;
                if (j < 4) {
                    sc = q[j] & 63;
                    mn = q[j + 4] & 63;
                } else {
                    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
                }
                wsc[r * WSC_LD + (sb >> 1) * 4 + (sb & 1)] = dm.x * static_cast<float>(sc);  // [jj][a lo, a hi, b lo, b hi]
                wsc[r * WSC_LD + (sb >> 1) * 4 + 2 + (sb & 1)] = dm.y * static_cast<float>(mn);
            }
        }
        if constexpr (DIAG != 6) __syncthreads();
        if (kt + S - 1 < kt0 + nk) issue(kt + S - 1);
        cp_async_commit();

        const uint8_t* ws = wraw(st);
        const int8_t* xs = xsm(st);
        const float* xc = xsc(st);
#pragma unroll (WN == 2 ? 2 : 1)  // 16 warps have 128 registers: one 64-weight step's fragments live at a time
        for (int jj = 0; jj < 2; ++jj) {  // 64-weight group: low nibbles = sub-block 2jj, high = 2jj + 1
            uint32_t alo[2][4], ahi[2][4];
            float ra[2][2][2], rb[2][2][2];  // [mt][row g / g+8][lo / hi]
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r = wm * 32 + mt * 16 + gq;
                const uint8_t* p = ws + r * P4_WS + 16 + jj * 32 + tig * 4;
                const uint32_t w[4] = {*reinterpret_cast<const uint32_t*>(p),
                                       *reinterpret_cast<const uint32_t*>(p + 8 * P4_WS),
                                       *reinterpret_cast<const uint32_t*>(p + 16),
                                       *reinterpret_cast<const uint32_t*>(p + 8 * P4_WS + 16)};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    alo[mt][i] = w[i] & 0x0F0F0F0Fu;
                    ahi[mt][i] = (w[i] >> 4) & 0x0F0F0F0Fu;
                }
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float4 w4 = *reinterpret_cast<const float4*>(wsc + (r + 8 * e) * WSC_LD + jj * 4);
                    ra[mt][e][0] = w4.x;
                    ra[mt][e][1] = w4.y;
                    rb[mt][e][0] = w4.z;
                    rb[mt][e][1] = w4.w;
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
                const int n = wn * TW + nt * 8;
                const int8_t* px = xs + (n + gq) * P4_XS + jj * 64 + tig * 4;
                const uint32_t blo[2] = {ldq32(px), ldq32(px + 16)};
                const uint32_t bhi[2] = {ldq32(px + 32), ldq32(px + 48)};
                const float* x0 = xc + (n + tig * 2) * 8;
                const float* x1 = x0 + 8;
#pragma unroll
                for (int nb = 0; nb < 2; ++nb) {  // lo / hi sub-block
                    const int blk = 2 * jj + nb;  // 32-block inside the stage
                    const float xd0 = x0[blk], xd1 = x1[blk], xs0 = x0[4 + blk], xs1 = x1[4 + blk];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c[4] = {0, 0, 0, 0};
                        if constexpr (DIAG == 4) {  // no mma: operands folded in so their loads stay
                            c[0] = static_cast<int>((nb ? ahi[mt][0] : alo[mt][0]) ^ (nb ? bhi[0] : blo[0]));
                            c[1] = static_cast<int>((nb ? ahi[mt][1] : alo[mt][1]) ^ (nb ? bhi[1] : blo[1]));
                            c[2] = static_cast<int>((nb ? ahi[mt][2] : alo[mt][2]));
                            c[3] = static_cast<int>((nb ? ahi[mt][3] : alo[mt][3]));
                        } else {
                            mma_s8_k32(c, nb ? ahi[mt] : alo[mt], nb ? bhi : blo);
                        }
                        float* o = acc[mt][nt];
                        if constexpr (DIAG == 1 || DIAG == 4) {
#pragma unroll
                            for (int e = 0; e < 4; ++e) o[e] += __int_as_float(c[e]);
                            continue;
                        } else if constexpr (DIAG == 2) {
                            o[0] = fmaf(ra[mt][0][nb] * xd0, i2f_small(c[0]), o[0]);
                            o[1] = fmaf(ra[mt][0][nb] * xd1, i2f_small(c[1]), o[1]);
                            o[2] = fmaf(ra[mt][1][nb] * xd0, i2f_small(c[2]), o[2]);
                            o[3] = fmaf(ra[mt][1][nb] * xd1, i2f_small(c[3]), o[3]);
                            continue;
                        }
                        o[0] = fmaf(ra[mt][0][nb] * xd0, i2f_small(c[0]), fmaf(-rb[mt][0][nb], xs0, o[0]));
                        o[1] = fmaf(ra[mt][0][nb] * xd1, i2f_small(c[1]), fmaf(-rb[mt][0][nb], xs1, o[1]));
                        o[2] = fmaf(ra[mt][1][nb] * xd0, i2f_small(c[2]), fmaf(-rb[mt][1][nb], xs0, o[2]));
                        o[3] = fmaf(ra[mt][1][nb] * xd1, i2f_small(c[3]), fmaf(-rb[mt][1][nb], xs1, o[3]));
                    }
                }
            }
        }
    }
    cp_async_wait<0>();

    if constexpr (GLU) {
        __half* Yh = reinterpret_cast<__half*>(Y);
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            const int r = r0 + wm * 16 + gq;
            const int c = c0 + wn * TW + nt * 8 + tig * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = r + (e >> 1) * 8;
                const int cc = c + (e & 1);
                const float g = acc[0][nt][e];
                const float sg = 1.f / (1.f + expf(-g));  // sigmoidf, as in k_silu_quant
                if (rr < rows && cc < T) Yh[static_cast<size_t>(cc) * ldy + rr] = __float2half(g * sg * acc[1][nt][e]);
            }
        }
        return;
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            const int r = r0 + wm * 32 + mt * 16 + gq;
            const int c = c0 + wn * TW + nt * 8 + tig * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = r + (e >> 1) * 8;
                const int cc = c + (e & 1);
                if (rr < rows && cc < T) Y[static_cast<size_t>(cc) * ldy + rr] = acc[mt][nt][e];
            }
        }
    }
}

// ================================================================================================
// Pipelined Q6_K int8 GEMM, same structure as k_mmq4. A 210-byte super-block b starts s = 2 b mod 16 bytes past a
// 16-byte boundary (rows are 16-aligned: K % 2048 == 0), uniform across the block for a stage. Each stage copies the
// three regions it needs per row and 128-wide K half h as 16-byte cp.async chunks from the 16-byte-aligned-down
// address, into windows of the row's smem slot:
//   ql[64 h .. +64) -> 5 chunks at P6_QL, qh[128 + 32 h .. +32) -> 3 chunks at P6_QH, scales + d [192 .. 210) ->
//   2 chunks at P6_SC (the last one with a partial source size, so nothing past the super-block is read).
// Logical byte L of a region is at window offset s + L; 4-byte words are read from s & ~3 with a 16-bit funnel
// shift when s & 3 == 2. This replaced 31 four-byte copies per row and stage, whose index math (div / mod by 31,
// branches) was ~20% of the kernel's stall samples in Nsight Compute.
// ================================================================================================

constexpr int P6_QL = 0, P6_QH = 80, P6_SC = 128;
// smem bytes per weight row and stage: 160 used. 176 = 44 words, so the 8 rows of a fragment load hit banks
// 12 gq + tig (mod 32), all different; 128 (32 words) gave an 8-way conflict on every weight word read (Nsight
// Compute: 344k of 393k wavefronts excess per load), 160 (40 words) would be 2-way.
constexpr int P6_WS = 176;

template <int BN>
__host__ __device__ constexpr size_t p6_stage_bytes() {
    return static_cast<size_t>(BM) * P6_WS + static_cast<size_t>(BN) * P4_XS + static_cast<size_t>(BN) * 8 * 4;
}
template <int BN, int S>
__host__ __device__ constexpr size_t p6_smem_bytes() {
    return S * p6_stage_bytes<BN>() + static_cast<size_t>(BM) * WSC_LD * 4;
}

// 16-byte cp.async of which only the first n bytes are read from global memory (the rest is zero-filled).
__device__ __forceinline__ void cp_async16_n(void* smem, const void* gmem, int n) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 4 : 0));
}

template <int BN, int S, int MB = 1, int DIAG = 0>  // DIAG: as k_mmq4 (1 = minimal epilogue, timing only)
__global__ void __launch_bounds__(NT, MB) k_mmq6(const uint8_t* W, const int8_t* xq, const float* xd,
                                               const float* xsd, float* Y, int rows, int K, int T, int ldy,
                                               int nsplit) {
    constexpr int TW = BN / 2;
    constexpr int NTW = TW / 8;
    constexpr size_t SB = p6_stage_bytes<BN>();
    extern __shared__ __align__(16) uint8_t p6_smem[];
    float* wsc = reinterpret_cast<float*>(p6_smem + S * SB);  // [BM][8]: d * scale of the stage's 8 16-groups
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gq = lane >> 2;
    const int tig = lane & 3;
    const int wm = warp >> 1;
    const int wn = warp & 1;
    const int c0 = blockIdx.x * BN;
    const int r0 = blockIdx.y * BM;
    const int KB = K / 32;
    const size_t row_bytes = static_cast<size_t>(K / 256) * 210;
    // Split-K: block z covers stages [kt0, kt0 + nk) and writes its partial sums to plane z of Y (k_splitk_sum
    // adds the planes in a fixed order, so results are deterministic).
    const int nk = K / P4_BK / nsplit;
    const int kt0 = static_cast<int>(blockIdx.z) * nk;
    Y += static_cast<size_t>(blockIdx.z) * T * ldy;

    auto wraw = [&](int st) { return p6_smem + st * SB; };
    auto xsm = [&](int st) { return reinterpret_cast<int8_t*>(p6_smem + st * SB + BM * P6_WS); };
    auto xsc = [&](int st) { return reinterpret_cast<float*>(p6_smem + st * SB + BM * P6_WS + BN * P4_XS); };

    auto issue = [&](int kt) {
        const int st = kt % S;
        const int k0 = kt * P4_BK;
        const int b = k0 >> 8;
        const int h = (k0 >> 7) & 1;
        const int s16 = (2 * b) & 15;  // super-block start mod 16
        uint8_t* ws = wraw(st);
        for (int i = tid; i < BM * 10; i += NT) {  // chunk c of row r: (c, r) = (i / BM, i % BM), BM a power of 2
            const int r = i & (BM - 1), c = i / BM;
            const bool ok = r0 + r < rows;
            const uint8_t* blk = W + (ok ? (r0 + r) * row_bytes : 0) + static_cast<size_t>(b) * 210;
            int src, dst, n = 16;
            if (c < 5) {
                src = h * 64 - s16 + 16 * c;
                dst = P6_QL + 16 * c;
            } else if (c < 8) {
                src = 128 + h * 32 - s16 + 16 * (c - 5);
                dst = P6_QH + 16 * (c - 5);
            } else {
                src = 192 - s16 + 16 * (c - 8);
                dst = P6_SC + 16 * (c - 8);
                if (c == 9) n = 2 + s16;  // stop at byte 210
            }
            cp_async16_n(ws + r * P6_WS + dst, blk + src, ok ? n : 0);
        }
        int8_t* xs = xsm(st);
        for (int i = tid; i < BN * (P4_BK / 16); i += NT) {
            const int n = i >> 3, ch = i & 7;
            const bool ok = c0 + n < T;
            cp_async16(xs + n * P4_XS + ch * 16, xq + (ok ? static_cast<size_t>(c0 + n) * K : 0) + k0 + ch * 16, ok);
        }
        float* sc = xsc(st);  // per token: xd of the stage's four 32-blocks (Q6_K has no min term: no xsd)
        for (int i = tid; i < BN; i += NT) {
            const bool ok = c0 + i < T;
            cp_async16(sc + i * 8, xd + (ok ? static_cast<size_t>(c0 + i) * KB : 0) + k0 / 32, ok);
        }
    };

    float acc[2][NTW][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NTW; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < nk) issue(kt0 + s);
        cp_async_commit();
    }
    for (int kt = kt0; kt < kt0 + nk; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        const int st = kt % S;
        const int h = ((kt * P4_BK) >> 7) & 1;
        const int s16 = (2 * ((kt * P4_BK) >> 8)) & 15;  // as in issue()
        const int sh = s16 & 3, s4 = s16 & ~3;
        const uint8_t* ws = wraw(st);
        for (int i = tid; i < BM * 8; i += NT) {  // d * scale of the 8 16-groups of this half
            const int r = i >> 3, g = i & 7;
            const uint8_t* s3 = ws + r * P6_WS + P6_SC + s16;  // logical byte 192
            const float d = __half2float(*reinterpret_cast<const __half*>(s3 + 16));
            // 16-group g = 2 q + lg -> [lg][q], so one 128-bit load gives a row's four scales for one lg
            wsc[r * WSC_LD + (g & 1) * 4 + (g >> 1)] = d * static_cast<float>(static_cast<int8_t>(s3[h * 8 + g]));
        }
        __syncthreads();
        if (kt + S - 1 < kt0 + nk) issue(kt + S - 1);
        cp_async_commit();

        const int8_t* xs = xsm(st);
        const float* xc = xsc(st);
        auto word = [&](const uint8_t* p) {  // logical 4-byte word at p - sh's region offset
            const uint32_t lo = *reinterpret_cast<const uint32_t*>(p);
            const uint32_t hi = *reinterpret_cast<const uint32_t*>(p + 4);
            return sh ? __funnelshift_r(lo, hi, 16) : lo;
        };
#pragma unroll
        for (int lg = 0; lg < 2; ++lg) {  // l in [16 lg, 16 lg + 16)
            uint32_t a[2][4][2];          // [mt][quarter][row g / g+8]
            float rs[2][2][4];            // [mt][row][quarter]
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int r = wm * 32 + mt * 16 + gq + 8 * e;
                    const uint8_t* rowp = ws + r * P6_WS;
                    const int l = lg * 16 + tig * 4;
                    const uint32_t qa = word(rowp + P6_QL + s4 + l), qb = word(rowp + P6_QL + s4 + 32 + l),
                                   qc = word(rowp + P6_QH + s4 + l);
                    // q - 32 per byte for q in [0, 63]: q + 0x60 cannot carry into the next byte, and flipping bit 7
                    // subtracts 128 mod 256 (2 ops; __vsub4 has no native instruction on sm_89 and is emulated)
                    a[mt][0][e] = (((qa & 0x0F0F0F0Fu) | ((qc << 4) & 0x30303030u)) + 0x60606060u) ^ 0x80808080u;
                    a[mt][1][e] = (((qb & 0x0F0F0F0Fu) | ((qc << 2) & 0x30303030u)) + 0x60606060u) ^ 0x80808080u;
                    a[mt][2][e] = ((((qa >> 4) & 0x0F0F0F0Fu) | (qc & 0x30303030u)) + 0x60606060u) ^ 0x80808080u;
                    a[mt][3][e] = ((((qb >> 4) & 0x0F0F0F0Fu) | ((qc >> 2) & 0x30303030u)) + 0x60606060u) ^ 0x80808080u;
                    const float4 w4 = *reinterpret_cast<const float4*>(wsc + r * WSC_LD + lg * 4);
                    rs[mt][e][0] = w4.x;
                    rs[mt][e][1] = w4.y;
                    rs[mt][e][2] = w4.z;
                    rs[mt][e][3] = w4.w;
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
                const int n = wn * TW + nt * 8;
                const float4 X0 = *reinterpret_cast<const float4*>(xc + (n + tig * 2) * 8);
                const float4 X1 = *reinterpret_cast<const float4*>(xc + (n + tig * 2 + 1) * 8);
                const float xa[4] = {X0.x, X0.y, X0.z, X0.w}, xb[4] = {X1.x, X1.y, X1.z, X1.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {  // quarter: k = 32 q + l
                    const uint32_t bq = ldq32(xs + (n + gq) * P4_XS + q * 32 + lg * 16 + tig * 4);
                    const float xd0 = xa[q], xd1 = xb[q];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c[4] = {0, 0, 0, 0};
                        mma_s8_k16(c, a[mt][q], bq);
                        float* o = acc[mt][nt];
                        if constexpr (DIAG == 1) {
#pragma unroll
                            for (int e = 0; e < 4; ++e) o[e] += __int_as_float(c[e]);
                            continue;
                        }
                        o[0] = fmaf(rs[mt][0][q] * xd0, i2f_small(c[0]), o[0]);
                        o[1] = fmaf(rs[mt][0][q] * xd1, i2f_small(c[1]), o[1]);
                        o[2] = fmaf(rs[mt][1][q] * xd0, i2f_small(c[2]), o[2]);
                        o[3] = fmaf(rs[mt][1][q] * xd1, i2f_small(c[3]), o[3]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            const int r = r0 + wm * 32 + mt * 16 + gq;
            const int c = c0 + wn * TW + nt * 8 + tig * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = r + (e >> 1) * 8;
                const int cc = c + (e & 1);
                if (rr < rows && cc < T) Y[static_cast<size_t>(cc) * ldy + rr] = acc[mt][nt][e];
            }
        }
    }
}

// Split-K reduction: Y[c * ldy + r] = sum over z = 0 .. nsplit - 1 of part[z][c][r] (planes of T x rows), in order.
__global__ void k_splitk_sum(const float* part, int nsplit, int rows, int T, float* Y, int ldy) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (r >= rows) return;
    const size_t plane = static_cast<size_t>(T) * rows;
    const float* p = part + static_cast<size_t>(c) * rows + r;
    float v = p[0];
    for (int z = 1; z < nsplit; ++z) v += p[z * plane];
    Y[static_cast<size_t>(c) * ldy + r] = v;
}

}  // namespace qgemm
}  // namespace akd
