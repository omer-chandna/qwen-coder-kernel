#pragma once

#include <adakernel/api.h>

#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

#include "qgemv.cuh"

namespace akd {

static constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float half_to_f(uint16_t bits) {
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sm[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        sm[warp] = v;
    }
    __syncthreads();
    const int nwarps = blockDim.x >> 5;
    float x = (threadIdx.x < nwarps) ? sm[threadIdx.x] : 0.f;
    if (warp == 0) {
        x = warp_sum(x);
        if (lane == 0) {
            sm[0] = x;
        }
    }
    __syncthreads();
    float out = sm[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float sm[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) {
        sm[warp] = v;
    }
    __syncthreads();
    const int nwarps = blockDim.x >> 5;
    float x = (threadIdx.x < nwarps) ? sm[threadIdx.x] : -1.0e30f;
    if (warp == 0) {
        x = warp_max(x);
        if (lane == 0) {
            sm[0] = x;
        }
    }
    __syncthreads();
    float out = sm[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ void rmsnorm_row(const float* x, const float* w, float* y, int n, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        ss += v * v;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float weight = w ? w[i] : 1.f;
        y[i] = x[i] * inv * weight;
    }
    __syncthreads();
}

__device__ __forceinline__ void residual_n(const float* a, const float* b, float* y, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = a[i] + b[i];
    }
    __syncthreads();
}

__device__ __forceinline__ void fused_add_rmsnorm_row(const float* x, const float* residual, const float* w,
                                                      float* y_add, float* y_norm, int n, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i] + residual[i];
        y_add[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float weight = w ? w[i] : 1.f;
        y_norm[i] = y_add[i] * inv * weight;
    }
    __syncthreads();
}

__device__ __forceinline__ void rope_neox(float* q, int n_heads, int head_dim, int pos, float theta) {
    const int half = head_dim >> 1;
    const int pairs = n_heads * half;
    for (int idx = threadIdx.x; idx < pairs; idx += blockDim.x) {
        const int h = idx / half;
        const int i = idx - h * half;
        const float freq = expf(-logf(theta) * (2.f * static_cast<float>(i) / static_cast<float>(head_dim)));
        const float ang = static_cast<float>(pos) * freq;
        const float c = cosf(ang);
        const float s = sinf(ang);
        float* row = q + static_cast<int>(h) * head_dim;
        const float x1 = row[i];
        const float x2 = row[half + i];
        row[i] = x1 * c - x2 * s;
        row[half + i] = x2 * c + x1 * s;
    }
    __syncthreads();
}

__device__ __forceinline__ void qk_norm(float* q, const float* w, int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; ++h) {
        rmsnorm_row(q + h * head_dim, w, q + h * head_dim, head_dim, eps);
    }
}

__device__ __forceinline__ void matvec_rows(const float* W, const float* x, float* y, int rows, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const bool vec = ((reinterpret_cast<uintptr_t>(W) | reinterpret_cast<uintptr_t>(x)) & 0xF) == 0 && (K & 3) == 0;
    for (int row = warp; row < rows; row += nwarps) {
        const float* wrow = W + static_cast<size_t>(row) * static_cast<size_t>(K);
        float acc = 0.f;
        if (vec) {
            const float4* w4 = reinterpret_cast<const float4*>(wrow);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            const int k4 = K >> 2;
            for (int j = lane; j < k4; j += 32) {
                const float4 a = w4[j];
                const float4 b = x4[j];
                acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int j = lane; j < K; j += 32) {
                acc += wrow[j] * x[j];
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            y[row] = acc;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void silu_mul_n(const float* gate, const float* up, float* y, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gate[i];
        const float sig = 1.f / (1.f + expf(-g));
        y[i] = g * sig * up[i];
    }
    __syncthreads();
}

struct AkKvView {
    int type;
    int page;
    int n_kv;
    int hd;
    int n_pages;
    int* pages;
    void* k;
    void* v;
    uint16_t* kscale;
    uint16_t* vscale;
};

__device__ __forceinline__ int kv_token_index(const AkKvView& view, int pos) {
    const int page = view.page > 0 ? view.page : 16;
    const int logical = pos / page;
    const int slot = pos - logical * page;
    const int phys = view.pages ? view.pages[logical] : logical;
    return phys * page + slot;
}

__device__ __forceinline__ float kv_load(const void* data, const uint16_t* scales, int type, int token, int n_kv, int hd,
                                         int kv_head, int elem) {
    const int index = (token * n_kv + kv_head) * hd + elem;
    if (type == 1) {
        return half_to_f(reinterpret_cast<const uint16_t*>(data)[index]);
    }
    if (type == 2) {
        const float scale = half_to_f(scales[token * n_kv + kv_head]);
        return scale * static_cast<float>(reinterpret_cast<const int8_t*>(data)[index]);
    }
    return reinterpret_cast<const float*>(data)[index];
}

__device__ __forceinline__ void kv_store_head(void* data, uint16_t* scales, int type, int token, int n_kv, int hd,
                                              int kv_head, const float* src) {
    const int base = (token * n_kv + kv_head) * hd;
    if (type == 1) {
        uint16_t* dst = reinterpret_cast<uint16_t*>(data) + base;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            dst[i] = __half_as_ushort(__float2half(src[kv_head * hd + i]));
        }
        return;
    }
    if (type == 2) {
        float amax = 0.f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            amax = fmaxf(amax, fabsf(src[kv_head * hd + i]));
        }
        amax = block_max(amax);
        const float scale = amax / 127.f;
        const float inv = scale != 0.f ? 1.f / scale : 0.f;
        int8_t* dst = reinterpret_cast<int8_t*>(data) + base;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            int q = static_cast<int>(rintf(src[kv_head * hd + i] * inv));
            q = max(-127, min(127, q));
            dst[i] = static_cast<int8_t>(q);
        }
        if (threadIdx.x == 0) {
            scales[token * n_kv + kv_head] = __half_as_ushort(__float2half(scale));
        }
        return;
    }
    float* dst = reinterpret_cast<float*>(data) + base;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        dst[i] = src[kv_head * hd + i];
    }
}

__device__ __forceinline__ void kv_write(const float* k, const float* v, float* kc, float* vc, const int* pos,
                                         int n_kv, int head_dim, int max_seq) {
    const int p = *pos;
    if (p >= 0 && p < max_seq) {
        const int n = n_kv * head_dim;
        float* kd = kc + static_cast<size_t>(p) * n;
        float* vd = vc + static_cast<size_t>(p) * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            kd[i] = k[i];
            vd[i] = v[i];
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void attn_head(const float* q, const float* kc, const float* vc, float* out, float* scores,
                                          int kv_head, int head_dim, int seq, int max_seq, int n_kv) {
    (void)max_seq;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        const float* krow = kc + (static_cast<size_t>(t) * n_kv + kv_head) * head_dim;
        float dot = 0.f;
        for (int i = 0; i < head_dim; ++i) {
            dot += q[i] * krow[i];
        }
        scores[t] = dot * rsqrtf(static_cast<float>(head_dim));
    }
    __syncthreads();
    float mx = -1.0e30f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        mx = fmaxf(mx, scores[t]);
    }
    mx = block_max(mx);
    float sum = 0.f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        const float e = expf(scores[t] - mx);
        scores[t] = e;
        sum += e;
    }
    sum = block_sum(sum);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        scores[t] *= inv;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < seq; ++t) {
            const float* vrow = vc + (static_cast<size_t>(t) * n_kv + kv_head) * head_dim;
            acc += scores[t] * vrow[i];
        }
        out[i] = acc;
    }
    __syncthreads();
}

__device__ __forceinline__ void kv_write_view(const float* k, const float* v, const int* pos, AkKvView view) {
    const int token = kv_token_index(view, *pos);
    for (int kv = 0; kv < view.n_kv; ++kv) {
        kv_store_head(view.k, view.kscale, view.type, token, view.n_kv, view.hd, kv, k);
        __syncthreads();
        kv_store_head(view.v, view.vscale, view.type, token, view.n_kv, view.hd, kv, v);
        __syncthreads();
    }
}

__device__ __forceinline__ void attn_head_view(const float* q, AkKvView view, float* out, float* scores, int kv_head,
                                              int seq) {
    const int head_dim = view.hd;
    const int n_kv = view.n_kv;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        float dot = 0.f;
        const int token = kv_token_index(view, t);
        for (int i = 0; i < head_dim; ++i) {
            dot += q[i] * kv_load(view.k, view.kscale, view.type, token, n_kv, head_dim, kv_head, i);
        }
        scores[t] = dot * rsqrtf(static_cast<float>(head_dim));
    }
    __syncthreads();
    float mx = -1.0e30f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        mx = fmaxf(mx, scores[t]);
    }
    mx = block_max(mx);
    float sum = 0.f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        const float e = expf(scores[t] - mx);
        scores[t] = e;
        sum += e;
    }
    sum = block_sum(sum);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        scores[t] *= inv;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < seq; ++t) {
            const int token = kv_token_index(view, t);
            acc += scores[t] * kv_load(view.v, view.vscale, view.type, token, n_kv, head_dim, kv_head, i);
        }
        out[i] = acc;
    }
    __syncthreads();
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ float q4k_at(const uint8_t* scales, const uint8_t* qs, float dsuper, float minsuper, int i) {
    const int group64 = i >> 6;
    const int inner = i & 63;
    const int is = group64 * 2;
    const uint8_t* q = qs + group64 * 32;
    uint8_t sc, m;
    if (inner < 32) {
        get_scale_min_k4(is, scales, &sc, &m);
        return dsuper * static_cast<float>(sc) * static_cast<float>(q[inner] & 0xF) - minsuper * static_cast<float>(m);
    }
    get_scale_min_k4(is + 1, scales, &sc, &m);
    return dsuper * static_cast<float>(sc) * static_cast<float>(q[inner - 32] >> 4) - minsuper * static_cast<float>(m);
}

__device__ __forceinline__ void q4k_matvec_rows(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nb = K >> 8;
    const int row_bytes = nb * 144;
    for (int row = warp; row < rows; row += nwarps) {
        const uint8_t* rowp = W + static_cast<size_t>(row) * row_bytes;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = rowp + b * 144;
            uint16_t d_bits, m_bits;
            memcpy(&d_bits, blk, 2);
            memcpy(&m_bits, blk + 2, 2);
            const float d = half_to_f(d_bits);
            const float mn = half_to_f(m_bits);
            const uint8_t* scales = blk + 4;
            const uint8_t* qs = blk + 16;
            const float* xb = x + b * 256;
            for (int i = lane; i < 256; i += 32) {
                acc += q4k_at(scales, qs, d, mn, i) * xb[i];
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            y[row] = acc;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float q6k_at(const uint8_t* blk, int i) {
    uint16_t d_bits;
    memcpy(&d_bits, blk + 208, sizeof(uint16_t));
    const float d = half_to_f(d_bits);
    const int half = i >> 7;
    const int local = i & 127;
    const uint8_t* ql = blk + half * 64;
    const uint8_t* qh = blk + 128 + half * 32;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + half * 8;
    int l = local;
    int q = 0;
    int scale_i = 0;
    if (local < 32) {
        l = local;
        q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
        scale_i = l >> 4;
    } else if (local < 64) {
        l = local - 32;
        q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
        scale_i = (l >> 4) + 2;
    } else if (local < 96) {
        l = local - 64;
        q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
        scale_i = (l >> 4) + 4;
    } else {
        l = local - 96;
        q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
        scale_i = (l >> 4) + 6;
    }
    return d * static_cast<float>(sc[scale_i]) * static_cast<float>(q - 32);
}

__device__ __forceinline__ void q6k_matvec_rows(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nb = K >> 8;
    const int row_bytes = nb * 210;
    for (int row = warp; row < rows; row += nwarps) {
        const uint8_t* rowp = W + static_cast<size_t>(row) * row_bytes;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = rowp + b * 210;
            const float* xb = x + b * 256;
            for (int i = lane; i < 256; i += 32) {
                acc += q6k_at(blk, i) * xb[i];
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            y[row] = acc;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void q8_matvec_rows(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nb = K >> 5;
    const int row_bytes = nb * 34;
    for (int row = warp; row < rows; row += nwarps) {
        const uint8_t* rowp = W + static_cast<size_t>(row) * row_bytes;
        float acc = 0.f;
        for (int b = lane; b < nb; b += 32) {
            const uint8_t* blk = rowp + b * 34;
            uint16_t d_bits;
            memcpy(&d_bits, blk, 2);
            const float d = half_to_f(d_bits);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            const float* xb = x + b * 32;
            float local = 0.f;
            for (int i = 0; i < 32; ++i) {
                local += static_cast<float>(qs[i]) * xb[i];
            }
            acc += local * d;
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            y[row] = acc;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void embed_row(const float* table, const int* token, float* x, int d) {
    const int t = *token;
    const float* src = table + static_cast<size_t>(t) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        x[i] = src[i];
    }
    __syncthreads();
}

// One row of a quantized embedding table, dequantized to float.
__device__ __forceinline__ void embed_q_row(const uint8_t* table, int type, const int* token, float* x, int d) {
    const int t = *token;
    if (type == 0) {
        const uint8_t* row = table + static_cast<size_t>(t) * (d >> 5) * 34;
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            const uint8_t* blk = row + (i >> 5) * 34;
            uint16_t bits;
            memcpy(&bits, blk, 2);
            x[i] = half_to_f(bits) * static_cast<float>(reinterpret_cast<const int8_t*>(blk + 2)[i & 31]);
        }
    } else if (type == 1) {
        const uint8_t* row = table + static_cast<size_t>(t) * (d >> 8) * 144;
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            const uint8_t* blk = row + (i >> 8) * 144;
            uint16_t db, mb;
            memcpy(&db, blk, 2);
            memcpy(&mb, blk + 2, 2);
            x[i] = q4k_at(blk + 4, blk + 16, half_to_f(db), half_to_f(mb), i & 255);
        }
    } else {
        const uint8_t* row = table + static_cast<size_t>(t) * (d >> 8) * 210;
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            x[i] = q6k_at(row + (i >> 8) * 210, i & 255);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ int argmax_block(const float* v, int n) {
    float best = -1.0e30f;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = v[i];
        if (x > best || (x == best && i < idx)) {
            best = x;
            idx = i;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ob = __shfl_down_sync(0xffffffff, best, off);
        const int oi = __shfl_down_sync(0xffffffff, idx, off);
        if (ob > best || (ob == best && oi < idx)) {
            best = ob;
            idx = oi;
        }
    }
    __shared__ float sb[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sb[warp] = best;
        si[warp] = idx;
    }
    __syncthreads();
    if (warp == 0) {
        const int nwarps = blockDim.x >> 5;
        best = (lane < nwarps) ? sb[lane] : -1.0e30f;
        idx = (lane < nwarps) ? si[lane] : 0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ob = __shfl_down_sync(0xffffffff, best, off);
            const int oi = __shfl_down_sync(0xffffffff, idx, off);
            if (ob > best || (ob == best && oi < idx)) {
                best = ob;
                idx = oi;
            }
        }
        if (lane == 0) {
            si[0] = idx;
        }
    }
    __syncthreads();
    const int out = si[0];
    __syncthreads();
    return out;
}

}  // namespace akd

#include "decode_ops.cuh"

namespace akd {

__device__ __forceinline__ int attn_seq(const ak_task& t) {
    int seq = t.p4 ? (*reinterpret_cast<const int*>(t.p4) + 1) : 1;
    if (seq < 1) seq = 1;
    if (t.n2 > 0 && seq > t.n2) seq = t.n2;
    return seq;
}

template <int T>
__device__ __forceinline__ void qgemv_dp4a_task(const ak_task& t) {
    const uint8_t* W = reinterpret_cast<const uint8_t*>(t.p0);
    const int8_t* xq = reinterpret_cast<const int8_t*>(t.p1);
    const float* xd = reinterpret_cast<const float*>(t.p3);
    float* y = reinterpret_cast<float*>(t.p2);
    if (t.p4) {
        qgemv_dp4a_rows<T>(W, xq, xd, y, t.n2, t.n1);
        qgemv_dp4a_rows<T>(reinterpret_cast<const uint8_t*>(t.p4), xq, xd, y + t.n2, t.n0 - t.n2, t.n1);
    } else {
        qgemv_dp4a_rows<T>(W, xq, xd, y, t.n0, t.n1);
    }
}

__device__ __forceinline__ void dispatch_task(const ak_task& t) {
    switch (t.op) {
        case AK_OP_RMSNORM:
            rmsnorm_row(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                        reinterpret_cast<float*>(t.p2), t.n0, t.f0);
            break;
        case AK_OP_RESIDUAL:
            residual_n(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                       reinterpret_cast<float*>(t.p2), t.n0);
            break;
        case AK_OP_FUSED_ADD_RMSNORM:
            fused_add_rmsnorm_row(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                                  reinterpret_cast<const float*>(t.p2), reinterpret_cast<float*>(t.p3),
                                  reinterpret_cast<float*>(t.p4), t.n0, t.f0);
            break;
        case AK_OP_ROPE: {
            const int pos = *reinterpret_cast<const int*>(t.p3);
            rope_neox(reinterpret_cast<float*>(t.p0), t.n0, t.n1, pos, t.f0);
            break;
        }
        case AK_OP_QK_NORM:
            qk_norm(reinterpret_cast<float*>(t.p0), reinterpret_cast<const float*>(t.p1), t.n0, t.n1, t.f0);
            break;
        case AK_OP_MATVEC:
            matvec_rows(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                        reinterpret_cast<float*>(t.p2), t.n0, t.n1);
            break;
        case AK_OP_Q8_MATVEC:
            q8_matvec_rows(reinterpret_cast<const uint8_t*>(t.p0), reinterpret_cast<const float*>(t.p1),
                           reinterpret_cast<float*>(t.p2), t.n0, t.n1);
            break;
        case AK_OP_Q4K_MATVEC:
            q4k_matvec_rows(reinterpret_cast<const uint8_t*>(t.p0), reinterpret_cast<const float*>(t.p1),
                            reinterpret_cast<float*>(t.p2), t.n0, t.n1);
            break;
        case AK_OP_Q6K_MATVEC:
            q6k_matvec_rows(reinterpret_cast<const uint8_t*>(t.p0), reinterpret_cast<const float*>(t.p1),
                            reinterpret_cast<float*>(t.p2), t.n0, t.n1);
            break;
        case AK_OP_QUANT_Q8_1:
            quantize_q8_1_n(reinterpret_cast<const float*>(t.p0), reinterpret_cast<int8_t*>(t.p1),
                            reinterpret_cast<float*>(t.p3), t.n0);
            break;
        case AK_OP_Q8_DP4A:
            qgemv_dp4a_task<0>(t);
            break;
        case AK_OP_Q4K_DP4A:
            qgemv_dp4a_task<1>(t);
            break;
        case AK_OP_Q6K_DP4A:
            qgemv_dp4a_task<2>(t);
            break;
        case AK_OP_SILU_MUL:
            silu_mul_n(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                       reinterpret_cast<float*>(t.p2), t.n0);
            break;
        case AK_OP_KV_WRITE:
            if (t.p6) {
                kv_write_view(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                              reinterpret_cast<const int*>(t.p4), *reinterpret_cast<const AkKvView*>(t.p6));
            } else {
                kv_write(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                         reinterpret_cast<float*>(t.p2), reinterpret_cast<float*>(t.p3),
                         reinterpret_cast<const int*>(t.p4), t.n0, t.n1, t.n2);
            }
            break;
        case AK_OP_ATTN_HEAD: {
            int seq = t.p4 ? (*reinterpret_cast<const int*>(t.p4) + 1) : 1;
            if (seq < 1) {
                seq = 1;
            }
            if (t.n2 > 0 && seq > t.n2) {
                seq = t.n2;
            }
            if (t.p6) {
                attn_head_view(reinterpret_cast<const float*>(t.p0), *reinterpret_cast<const AkKvView*>(t.p6),
                               reinterpret_cast<float*>(t.p3), reinterpret_cast<float*>(t.p5), t.n0, seq);
            } else {
                attn_head(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                          reinterpret_cast<const float*>(t.p2), reinterpret_cast<float*>(t.p3),
                          reinterpret_cast<float*>(t.p5), t.n0, t.n1, seq, t.n2, t.n3);
            }
            break;
        }
        case AK_OP_EMBED:
            embed_row(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const int*>(t.p1),
                      reinterpret_cast<float*>(t.p2), t.n0);
            break;
        case AK_OP_EMBED_Q:
            embed_q_row(reinterpret_cast<const uint8_t*>(t.p0), t.i0, reinterpret_cast<const int*>(t.p1),
                        reinterpret_cast<float*>(t.p2), t.n0);
            break;
        case AK_OP_ADD_NORM_QUANT:
            add_rmsnorm_quant(reinterpret_cast<float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                              reinterpret_cast<const float*>(t.p2), reinterpret_cast<int8_t*>(t.p4),
                              reinterpret_cast<float*>(t.p3), t.n0, t.f0);
            break;
        case AK_OP_SILU_QUANT:
            silu_quant(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                       reinterpret_cast<int8_t*>(t.p2), reinterpret_cast<float*>(t.p3), t.n0);
            break;
        case AK_OP_QK_NORM_ROPE:
            qk_norm_rope(reinterpret_cast<float*>(t.p0), reinterpret_cast<const float*>(t.p1), t.n0, t.n1, t.f0,
                         reinterpret_cast<const int*>(t.p3), __int_as_float(t.i0), reinterpret_cast<kv_t*>(t.p2),
                         reinterpret_cast<const float*>(t.p4), reinterpret_cast<kv_t*>(t.p5), t.n2);
            break;
        case AK_OP_ATTN_DECODE:
            attn_decode_head(reinterpret_cast<const float*>(t.p0), reinterpret_cast<const float*>(t.p1),
                             reinterpret_cast<const float*>(t.p2), reinterpret_cast<int8_t*>(t.p3),
                             reinterpret_cast<float*>(t.p6), reinterpret_cast<float*>(t.p5), t.n0, t.n1, attn_seq(t),
                             t.n3);
            break;
        case AK_OP_ATTN_PART:
            attn_part(reinterpret_cast<const float*>(t.p0), 0, reinterpret_cast<const kv_t*>(t.p1),
                      reinterpret_cast<const kv_t*>(t.p2), reinterpret_cast<float*>(t.p3), t.n0, t.n3, t.n1, t.n2, t.i0,
                      t.i1 & 0xFF, reinterpret_cast<const int*>(t.p4), (t.i1 >> 8) & 0xFF, (t.i1 >> 16) & 0xFF, 0);
            break;
        case AK_OP_ATTN_COMBINE:
            attn_combine(reinterpret_cast<const float*>(t.p3), t.n0, t.n1, t.i1 & 0xFF, t.i0, t.n2,
                         reinterpret_cast<const int*>(t.p4), reinterpret_cast<int8_t*>(t.p5),
                         reinterpret_cast<float*>(t.p6), (t.i1 >> 8) & 0xFF, 0);
            break;
        case AK_OP_ARGMAX_PART:
            argmax_part(reinterpret_cast<const float*>(t.p0), t.n0, t.i0, reinterpret_cast<unsigned long long*>(t.p1));
            break;
        case AK_OP_FINALIZE: {
            const int sample = t.p4 ? *reinterpret_cast<const int*>(t.p4) : 1;
            int id = 0;
            if (t.p6) {
                // Packed result from AK_OP_ARGMAX_PART tasks; clear it for the next step.
                __shared__ int packed_id;
                if (threadIdx.x == 0) {
                    unsigned long long* best = reinterpret_cast<unsigned long long*>(t.p6);
                    packed_id = argmax_key_index(*reinterpret_cast<volatile unsigned long long*>(best));
                    *best = 0ull;
                }
                __syncthreads();
                id = packed_id;
            } else {
                id = argmax_block(reinterpret_cast<const float*>(t.p0), t.n0);
            }
            if (threadIdx.x == 0) {
                int* pos = reinterpret_cast<int*>(t.p1);
                int* token = reinterpret_cast<int*>(t.p2);
                int* out = reinterpret_cast<int*>(t.p3);
                int* count = reinterpret_cast<int*>(t.p5);
                if (pos) {
                    *pos = *pos + 1;
                }
                if (sample && count && out && token) {
                    const int c = *count;
                    out[c] = id;
                    *count = c + 1;
                    *token = id;
                }
            }
            __syncthreads();
            break;
        }
        default:
            __syncthreads();
            break;
    }
}

}  // namespace akd
