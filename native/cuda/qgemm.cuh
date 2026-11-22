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

// Raw bytes for one thread's half weight row of a K tile (same fetch as the fp16 kernel), then integer decode.
template <int type>
__device__ __forceinline__ void store_w8(const Raw& R, int8_t* Wq, float* wsc, int row, int half, int k0) {
    const int off = half * 16;
    uint4 lo, hi;
    if (type == 1) {
        lo = make_uint4(R.r[0] & 0x0F0F0F0Fu, R.r[1] & 0x0F0F0F0Fu, R.r[2] & 0x0F0F0F0Fu, R.r[3] & 0x0F0F0F0Fu);
        hi = make_uint4((R.r[0] >> 4) & 0x0F0F0F0Fu, (R.r[1] >> 4) & 0x0F0F0F0Fu, (R.r[2] >> 4) & 0x0F0F0F0Fu,
                        (R.r[3] >> 4) & 0x0F0F0F0Fu);
        if (half == 0) {
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
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&dm));
            *reinterpret_cast<float4*>(wsc + row * 4) =
                make_float4(f.x * static_cast<float>(sc0), f.x * static_cast<float>(sc1), f.y * static_cast<float>(m0),
                            f.y * static_cast<float>(m1));
        }
    } else {
        const int odd = ((k0 & 255) >> 6) & 1;
        uint32_t qa[4], qb[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t a = R.r[w], b = R.r[4 + w], c = R.r[8 + w];
            const uint32_t v0 = odd ? (((a >> 4) & 0x0F0F0F0Fu) | (c & 0x30303030u))
                                    : ((a & 0x0F0F0F0Fu) | ((c << 4) & 0x30303030u));
            const uint32_t v1 = odd ? (((b >> 4) & 0x0F0F0F0Fu) | ((c >> 2) & 0x30303030u))
                                    : ((b & 0x0F0F0F0Fu) | ((c << 2) & 0x30303030u));
            qa[w] = __vsub4(v0, 0x20202020u);
            qb[w] = __vsub4(v1, 0x20202020u);
        }
        lo = make_uint4(qa[0], qa[1], qa[2], qa[3]);
        hi = make_uint4(qb[0], qb[1], qb[2], qb[3]);
        const int8_t sa = static_cast<int8_t>(R.r[12] & 0xFFu);
        const int8_t sb = static_cast<int8_t>((R.r[12] >> 8) & 0xFFu);
        const uint16_t dbits = static_cast<uint16_t>(R.r[12] >> 16);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dbits));
        // groups of 16 in this tile: k 0-15, 16-31 (first quarter), 32-47, 48-63 (second); this thread owns `half`
        wsc[row * 4 + half] = d * static_cast<float>(sa);
        wsc[row * 4 + 2 + half] = d * static_cast<float>(sb);
    }
    *reinterpret_cast<uint4*>(Wq + row * LDQ + off) = lo;
    *reinterpret_cast<uint4*>(Wq + row * LDQ + 32 + off) = hi;
}

// grid (ceil(T / BN), ceil(rows / BM)). xq / xd / xsd from k_quant_x8 on the same K.
template <int type, int BN>
__global__ void __launch_bounds__(NT, 2) k_qgemm8(const uint8_t* W, const int8_t* xq, const float* xd, const float* xsd,
                                               float* Y, int rows, int K, int T, int ldy) {
    static_assert(BN == 64 || BN == 128, "token tile");
    __shared__ __align__(16) int8_t Wq[BM * LDQ];
    __shared__ __align__(16) int8_t Xq[BN * LDQ];
    __shared__ __align__(16) float wsc[BM * 4];  // Q4_K: a0 a1 b0 b1; Q6_K: s of the four 16-groups
    __shared__ __align__(16) float xsc[BN * 4];  // xd0 xd1 xsd0 xsd1
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gq = lane >> 2;
    const int tig = lane & 3;
    const int c0 = blockIdx.x * BN;
    const int r0 = blockIdx.y * BM;
    const int wrow = tid >> 1;
    const int whalf = tid & 1;
    const int wm = warp >> 1;
    const int wn = warp & 1;
    const int KB = K / 32;

    Raw R;
    constexpr int XV = BN * 4 / NT;  // uint4 of int8 activations per thread per tile
    uint4 xr[XV];
    float4 xs4 = make_float4(0.f, 0.f, 0.f, 0.f);
    auto load_x = [&](int k0) {
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int idx = tid + i * NT;
            const int n = idx >> 2;
            const int c = c0 + n;
            xr[i] = c < T ? *reinterpret_cast<const uint4*>(xq + static_cast<size_t>(c) * K + k0 + (idx & 3) * 16)
                          : make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid < BN) {
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
            *reinterpret_cast<uint4*>(Xq + (idx >> 2) * LDQ + (idx & 3) * 16) = xr[i];
        }
        if (tid < BN) *reinterpret_cast<float4*>(xsc + tid * 4) = xs4;
    };

    float acc[2][BN / 16][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < BN / 16; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

    const int nk = K / BK;
    load_raw<type>(R, W, rows, K, r0 + wrow, whalf, 0);
    load_x(0);
    store_w8<type>(R, Wq, wsc, wrow, whalf, 0);
    store_x();
    __syncthreads();
    for (int kt = 0; kt < nk; ++kt) {
        const int k_next = (kt + 1) * BK;
        if (kt + 1 < nk) {
            load_raw<type>(R, W, rows, K, r0 + wrow, whalf, k_next);
            load_x(k_next);
        }
        if (type == 1) {
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
                for (int nt = 0; nt < BN / 16; ++nt) {
                    const int n = wn * (BN / 2) + nt * 8;
                    const int8_t* p = Xq + (n + gq) * LDQ + sb * 32 + tig * 4;
                    const uint32_t b[2] = {ldq32(p), ldq32(p + 16)};
                    const float xd0 = xsc[(n + tig * 2) * 4 + sb], xd1 = xsc[(n + tig * 2 + 1) * 4 + sb];
                    const float xs0 = xsc[(n + tig * 2) * 4 + 2 + sb], xs1 = xsc[(n + tig * 2 + 1) * 4 + 2 + sb];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c[4] = {0, 0, 0, 0};
                        mma_s8_k32(c, a[mt], b);
                        float* o = acc[mt][nt];
                        o[0] += ra[mt][0] * xd0 * static_cast<float>(c[0]) - rb[mt][0] * xs0;
                        o[1] += ra[mt][0] * xd1 * static_cast<float>(c[1]) - rb[mt][0] * xs1;
                        o[2] += ra[mt][1] * xd0 * static_cast<float>(c[2]) - rb[mt][1] * xs0;
                        o[3] += ra[mt][1] * xd1 * static_cast<float>(c[3]) - rb[mt][1] * xs1;
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
                for (int nt = 0; nt < BN / 16; ++nt) {
                    const int n = wn * (BN / 2) + nt * 8;
                    const uint32_t b = ldq32(Xq + (n + gq) * LDQ + g16 * 16 + tig * 4);
                    const float xd0 = xsc[(n + tig * 2) * 4 + (g16 >> 1)];
                    const float xd1 = xsc[(n + tig * 2 + 1) * 4 + (g16 >> 1)];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c[4] = {0, 0, 0, 0};
                        mma_s8_k16(c, a[mt], b);
                        float* o = acc[mt][nt];
                        o[0] += rs[mt][0] * xd0 * static_cast<float>(c[0]);
                        o[1] += rs[mt][0] * xd1 * static_cast<float>(c[1]);
                        o[2] += rs[mt][1] * xd0 * static_cast<float>(c[2]);
                        o[3] += rs[mt][1] * xd1 * static_cast<float>(c[3]);
                    }
                }
            }
        }
        __syncthreads();
        if (kt + 1 < nk) {
            store_w8<type>(R, Wq, wsc, wrow, whalf, k_next);
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

}  // namespace qgemm
}  // namespace akd
