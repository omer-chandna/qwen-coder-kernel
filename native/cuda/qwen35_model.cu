// Qwen3.5 ("qwen35" GGUF architecture) hybrid decoder: Gated DeltaNet linear attention in three of every four
// layers, gated full attention in the fourth, plus the file's MTP (multi-token prediction) head used as a built-in
// speculative draft model. Runs straight from a llama.cpp GGUF file.
//
// Matrices stay in their file format (Q4_K, Q6_K, Q8_0) and go through int8-activation dp4a GEMVs (qgemv.cuh).
// Everything between the GEMVs is here:
//
//   linear-attention layer   x -> RMSNorm -> [qkv | z] GEMV, [alpha | beta] GEMV
//                            -> causal conv1d (kernel 4, rolling state) + SiLU over the qkv channels
//                            -> per value head: L2-norm q, k; S = exp(g) S; d = beta (v - S^T k); S += k d^T;
//                               o = S^T q / sqrt(128)   (state 128 x 128 fp32 per head)
//                            -> RMSNorm(o) * w * SiLU(z) -> out GEMV
//   full-attention layer     x -> RMSNorm -> [q | gate] (per head), k, v GEMVs -> q/k RMSNorm, NeoX RoPE on the
//                            first 64 of 256 dims -> split-key GQA attention over an fp16 cache
//                            -> * sigmoid(gate) -> out GEMV
//   every layer              residual, post-attention RMSNorm, SwiGLU FFN, residual
//
// Every kernel takes T token columns at consecutive positions. Decode and speculative verify (T <= kMaxCols) feed
// int8 (Q8_1) activations to dp4a GEMVs; qgemv_dp4a_rows_mc keeps each column's summation order, so a column's
// result is the single-column result. Long prompts go through the same kernels in chunks of up to kPfCols tokens
// with fp16 activations and cuBLAS GEMMs against weights expanded to fp16 once per chunk.
//
// Recurrent state rollback: a decode / verify pass never writes the state of its own columns. It saves their inputs
// (raw conv inputs; normalized k, v, decay, beta) into one of two buffers, and the next pass first replays the
// ctl.commit accepted columns from the other buffer, with the same arithmetic, before running its own. Accepting n
// of T verified tokens is just commit = n. Prefill passes (write mode) run all their columns into the state.
//
// MTP head (blk.N, one full-attention block): input pair (h_p, token_{p+1}) at MTP position p + 1, where h_p is
// the target's final-normed hidden at p. e = RMSNorm(embed(token)), h = RMSNorm(h_p); x = eh_proj [e | h]; one
// attention block + FFN with its own KV cache; RMSNorm (shared_head_norm) -> LM head -> draft token. Further drafts
// chain through the MTP's own normed output, as llama.cpp does.
//
// The token embedding table lives in pinned host memory (one row per token over PCIe).

#include "dequant.cuh"
#include "qgemm.cuh"
#include "error.h"
#include "launch.h"
#include "ops.cuh"

#include <adakernel/api.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

enum { kF32 = 0, kQ8 = 8, kQ4K = 12, kQ6K = 14 };

constexpr int kDk = 128;         // linear-attention head size (key and value)
constexpr int kHd = 256;         // full-attention head size
constexpr int kAttnChunk = 256;  // keys per split-attention block
constexpr int kMaxCols = 12;     // widest decode / verify pass: 1 + 11 drafts
constexpr int kSub = 8;          // delta-rule columns staged in shared memory at a time
constexpr int kPfCols = 1024;    // default prefill chunk (AK_Q35_PF_COLS overrides); 512 -> 1024 cut TTFT 14-22% on coding prompts
constexpr int kPfMin = 9;        // ranges of <= 9 tokens go through the GEMV path (measured crossover: GEMV 9 / 10 / 12
                                 // columns 36 / 38 / 45 ms, a 13-token GEMM chunk 38 ms; it was 24, so 13-24 tokens took two
                                 // GEMV passes, >= 67 ms)

// Recurrent-state snapshots taken inside a pass: after column col[i], k_conv / k_gdn also store the state of each
// linear layer into base[i] (a prefix-cache slot, laid out as snap_copy does: per linear layer [conv 3 x C | gdn]).
// This lets a prompt be processed in one batched pass even when the prefix cache wants the state at positions in
// the middle of it (docs/QWEN38_9B.md, "Snapshots inside a chunk").
struct SnapReq {
    float* base[AK_Q35_CACHE_SLOTS];
    int col[AK_Q35_CACHE_SLOTS];
    int n;
};

// ------------------------------------------------------------------------------------------------
// GGUF header
// ------------------------------------------------------------------------------------------------

struct Tensor {
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;  // in the file's data section
};

struct File {
    std::map<std::string, double> num;
    std::map<std::string, std::string> str;
    std::map<std::string, Tensor> tensors;
    uint64_t data_offset = 0;
};

struct Rd {
    std::ifstream in;
    template <class T>
    T get() {
        T v{};
        in.read(reinterpret_cast<char*>(&v), sizeof(T));
        return v;
    }
    std::string s() {
        const uint64_t n = get<uint64_t>();
        if (n > (1ull << 30)) {
            in.setstate(std::ios::failbit);
            return {};
        }
        std::string r(static_cast<size_t>(n), '\0');
        if (n) in.read(r.data(), static_cast<std::streamsize>(n));
        return r;
    }
    // Scalars come back as double, strings via *str; arrays are skipped.
    int value(uint32_t t, double* d, std::string* str) {
        switch (t) {
            case 0: *d = get<uint8_t>(); return 1;
            case 1: *d = get<int8_t>(); return 1;
            case 2: *d = get<uint16_t>(); return 1;
            case 3: *d = get<int16_t>(); return 1;
            case 4: *d = get<uint32_t>(); return 1;
            case 5: *d = get<int32_t>(); return 1;
            case 6: *d = get<float>(); return 1;
            case 7: *d = get<uint8_t>(); return 1;
            case 8: *str = s(); return 2;
            case 9: {
                const uint32_t et = get<uint32_t>();
                const uint64_t n = get<uint64_t>();
                double dd;
                std::string ss;
                for (uint64_t i = 0; i < n && in; ++i) value(et, &dd, &ss);
                return 0;
            }
            case 10: *d = static_cast<double>(get<uint64_t>()); return 1;
            case 11: *d = static_cast<double>(get<int64_t>()); return 1;
            case 12: *d = get<double>(); return 1;
            default: in.setstate(std::ios::failbit); return 0;
        }
    }
};

ak_status parse(const char* path, File* f) {
    Rd r;
    r.in.open(path, std::ios::binary);
    if (!r.in) {
        ak_set_error("qwen35: cannot open file");
        return AK_ERR_INVALID;
    }
    if (r.get<uint32_t>() != 0x46554747u || r.get<uint32_t>() < 2) {
        ak_set_error("qwen35: not a GGUF v2+ file");
        return AK_ERR_INVALID;
    }
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();
    for (uint64_t i = 0; i < n_kv && r.in; ++i) {
        const std::string key = r.s();
        double d = 0;
        std::string s;
        const int kind = r.value(r.get<uint32_t>(), &d, &s);
        if (kind == 1) f->num[key] = d;
        if (kind == 2) f->str[key] = s;
    }
    for (uint64_t i = 0; i < n_tensors && r.in; ++i) {
        const std::string name = r.s();
        Tensor t;
        const uint32_t nd = r.get<uint32_t>();
        if (nd > 4) {
            ak_set_error("qwen35: tensor with more than 4 dims");
            return AK_ERR_INVALID;
        }
        for (uint32_t k = 0; k < nd; ++k) t.dims.push_back(r.get<uint64_t>());
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        f->tensors[name] = t;
    }
    if (!r.in) {
        ak_set_error("qwen35: truncated header");
        return AK_ERR_INVALID;
    }
    uint64_t align = 32;
    auto it = f->num.find("general.alignment");
    if (it != f->num.end() && it->second > 0) align = static_cast<uint64_t>(it->second);
    const uint64_t end = static_cast<uint64_t>(r.in.tellg());
    f->data_offset = (end + align - 1) / align * align;
    return AK_OK;
}

size_t tensor_bytes(const Tensor& t) {
    size_t n = 1;
    for (uint64_t d : t.dims) n *= d;
    switch (t.type) {
        case kF32: return n * 4;
        case kQ8: return n / 32 * 34;
        case kQ4K: return n / 256 * 144;
        case kQ6K: return n / 256 * 210;
        default: return 0;
    }
}

int qcode(uint32_t t) { return t == kQ8 ? 0 : (t == kQ4K ? 1 : (t == kQ6K ? 2 : -1)); }

struct QMat {
    const uint8_t* w = nullptr;
    int type = -1;  // 0 Q8_0, 1 Q4_K, 2 Q6_K
    int rows = 0;
    int K = 0;
    size_t bytes() const {
        const int nb = type == 0 ? K / 32 : K / 256;
        const int bb = type == 0 ? 34 : (type == 1 ? 144 : 210);
        return static_cast<size_t>(nb) * bb * rows;
    }
};

struct Layer {
    bool linear = true;
    const float* attn_norm = nullptr;
    const float* post_norm = nullptr;
    // linear attention
    QMat wqkv, wz, walpha, wbeta, wout;
    const float* conv_w = nullptr;  // [channels][4]
    const float* dt_bias = nullptr;
    const float* ssm_a = nullptr;   // -exp(A_log)
    const float* ssm_norm = nullptr;
    float* conv_state = nullptr;    // [3][channels], oldest first
    float* gdn_state = nullptr;     // [Hv][Dv][Dk]
    float* conv_save = nullptr;     // [2][kMaxCols][channels]   raw conv inputs of the last pass
    float* k_save = nullptr;        // [2][kMaxCols][Hv][Dk]     normalized keys (per value head)
    float* v_save = nullptr;        // [2][kMaxCols][Hv][Dv]
    float* db_save = nullptr;       // [2][kMaxCols][2 Hv]       decay | beta
    // full attention
    QMat wq, wk, wv, wo;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    __half* kc = nullptr;  // [max_seq][Hkv][hd]
    __half* vc = nullptr;
    // FFN
    QMat wgate, wup, wdown;
};

struct Mtp {
    bool present = false;
    Layer blk;
    QMat eh_proj;  // [d rows][2d]
    const float* enorm = nullptr;
    const float* hnorm = nullptr;
    const float* head_norm = nullptr;
};

ak_status ck(cudaError_t e) {
    if (e != cudaSuccess) {
        ak_set_error(cudaGetErrorString(e));
        return e == cudaErrorMemoryAllocation ? AK_ERR_OOM : AK_ERR_CUDA;
    }
    return AK_OK;
}

// Profile buckets (ak_q35_profile).
enum Kind {
    kKEmbed, kKNorm, kKGemvQkv, kKGemvAb, kKConv, kKGdn, kKGatedNorm, kKGemvOut, kKGemvAttnIn, kKAttnPrep,
    kKAttnPart, kKAttnCombine, kKGemvAttnOut, kKGemvGateUp, kKSilu, kKGemvDown, kKGemvLogits, kKArgmax, kKCount
};

// Device control block: target position of column 0, saved columns to replay first, save buffer the previous pass
// wrote (this pass writes the other), MTP position of column 0.
enum { kCtlPos = 0, kCtlCommit = 1, kCtlPar = 2, kCtlMpos = 3, kCtlN = 4 };

// Activation buffers for one pass width. Decode buffers (cols = kMaxCols) use Q8_1 activations; prefill buffers
// (cols = pf_cols) use fp16 activations in a16, whose column stride is the consuming matrix's K.
struct Bufs {
    int cols = 0;
    bool pf = false;
    float *x = nullptr, *proj = nullptr, *ebuf = nullptr, *hbuf = nullptr, *mh = nullptr;
    float *big = nullptr, *ab = nullptr, *conv_out = nullptr, *o = nullptr, *q = nullptr, *v = nullptr, *gu = nullptr;
    int8_t *hq_q = nullptr, *aq = nullptr, *fq = nullptr;
    float *hq_d = nullptr, *ad = nullptr, *fd = nullptr;
    __half* a16 = nullptr;
    int* tok = nullptr;
    int* mtok = nullptr;
};

}  // namespace

struct ak_q35_model {
    ak_q35_info info{};
    int n_layers = 0, d = 0, ff = 0, vocab = 0, max_seq = 0;
    int hq = 0, hkv = 0, n_rot = 0;
    int hk = 0, hv = 0, conv_ch = 0, d_inner = 0;
    int big_n = 0;  // per-column stride of `big`
    int aq_n = 0;   // per-column stride of `aq`
    float eps = 1e-6f, theta = 1e7f;
    uint8_t* blob = nullptr;
    uint8_t* embd_host = nullptr;  // pinned, mapped
    const uint8_t* embd_dev = nullptr;
    int embd_type = 1;
    std::vector<Layer> layers;
    Mtp mtp;
    QMat out;
    const float* out_norm = nullptr;
    Bufs dec;                            // decode / verify, kMaxCols columns
    Bufs pf;                             // prefill, pf_cols columns, allocated on the first long prompt
    int pf_cols = kPfCols;
    bool no_pf = false;                  // GEMV-only prefill (ak_q35_set_prefill 0, or AK_Q35_NO_PF=1)
    int attn = 2;                        // attention kernels: 0 per-column scalar, 1 k_attn_mma, 2 k_attn_mma2
                                         // (AK_Q35_ATTN; AK_Q35_ATTN_OLD=1 is 0)
    int qgemm_max = 768;                 // prefill chunks up to this many columns use the fused quantized GEMM
                                         // (AK_Q35_QGEMM_MAX; 0 = always dequant + cuBLAS)
    bool int8_gemm = true;               // prefill GEMMs on int8 tensor cores with Q8_1 activations, the decode path's
                                         // arithmetic (AK_Q35_INT8=0: fp16 fused kernel / cuBLAS)
    int8_t* xq8 = nullptr;               // int8 prefill activations [pf_cols][max K]
    float *xd8 = nullptr, *xsd8 = nullptr;  // their per-32 scales and scale * sum
    int int8_max = 1 << 30;              // longest chunk the int8 path takes (AK_Q35_INT8_MAX)
    int mmq_short = 0, mmq_long = 2;     // pipelined int8 GEMM configurations (mmq_dispatch)
    float* splitk = nullptr;             // split-K partial sums of the int8 GEMMs [nsplit][T][rows]
    size_t splitk_n = 0;                 // its size in floats
    int splitk_max = 4;                  // most K splits per int8 GEMM (AK_Q35_SPLITK; 1 = off)
    int n_sm = 0;                        // streaming multiprocessors on the device
    SnapReq snapreq{};                   // snapshots the next target pass takes (n = 0: none)
    int qdiag = 0;                       // AK_Q35_QDIAG: 1 = the plain int8 kernel skips the math (timing only),
                                         // 2 = Q4_K uses the plain int8 kernel instead of the pipelined one
    int draft_vocab = 0;                 // rows of the LM head the MTP draft scores (a prefix of the vocab)
    __half* w16 = nullptr;               // one matrix expanded to fp16 (prefill)
    size_t w16_n = 0;
    cublasHandle_t blas = nullptr;
    int* h_pf = nullptr;                 // pinned prefill token staging [2][pf_cols]
    float *logits = nullptr, *part = nullptr;
    unsigned long long* best = nullptr;  // kMaxCols + 1 (last: MTP)
    int *d_arg = nullptr, *d_ctl = nullptr, *d_out = nullptr, *d_count = nullptr;
    int* h_pin = nullptr;  // pinned staging: [ctl kCtlN][tok kMaxCols][mtok kMaxCols][results 2 kMaxCols]
    int n_split = 1;
    float* snap[AK_Q35_CACHE_SLOTS] = {};  // recurrent-state snapshots (conv | gdn per linear layer), lazily allocated
    std::vector<void*> mem;
    size_t dev_bytes = 0;
    size_t weight_bytes = 0;
    size_t mtp_bytes = 0;
    cudaStream_t stream = nullptr;
    std::map<std::vector<uint64_t>, cudaGraphExec_t> graphs;
    float capture_ms = 0.f;
};

namespace {

ak_status dalloc(ak_q35_model* m, void** p, size_t bytes) {
    *p = nullptr;
    if (!bytes) return AK_OK;
    ak_status st = ck(cudaMalloc(p, bytes));
    if (st != AK_OK) return st;
    m->mem.push_back(*p);
    m->dev_bytes += bytes;
    return AK_OK;
}

template <class T>
ak_status dalloc_n(ak_q35_model* m, T** p, size_t n) {
    void* raw = nullptr;
    ak_status st = dalloc(m, &raw, n * sizeof(T));
    *p = static_cast<T*>(raw);
    return st;
}

// ------------------------------------------------------------------------------------------------
// Kernels. Column c of a T-column pass is blockIdx.y (or .z where noted).
// ------------------------------------------------------------------------------------------------

__device__ __forceinline__ float xor_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// One warp's 32 values -> one Q8_1 block, or fp16 into o16 (prefill). Warp-uniform branch.
__device__ __forceinline__ void quant32(float v, int8_t* xq, float* xd, __half* o16, int idx) {
    if (o16) {
        o16[idx] = __float2half(v);
        return;
    }
    const float amax = akd::qg_warp_max(fabsf(v));
    const float dd = amax / 127.f;
    xq[idx] = static_cast<int8_t>(amax > 0.f ? __float2int_rn(v / dd) : 0);
    if ((threadIdx.x & 31) == 0) xd[idx >> 5] = dd;
}

// Block-wide: x += res (if res); y = RMSNorm(x) * w -> Q8_1 (xq, xd) or fp16 o16, and to hout if set. n % 32 == 0.
__device__ __forceinline__ void norm_quant(float* x, const float* res, const float* w, int8_t* xq, float* xd,
                                           __half* o16, int n, float eps, float* hout) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (res) {
            v += res[i];
            x[i] = v;
        }
        ss += v * v;
    }
    ss = akd::block_sum(ss);  // syncs, so the x writes above are visible below
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    for (int base = (threadIdx.x >> 5) * 32; base < n; base += nwarps * 32) {
        const float v = x[base + lane] * inv * w[base + lane];
        if (hout) hout[base + lane] = v;
        quant32(v, xq, xd, o16, base + lane);
    }
    __syncthreads();
}

__global__ void k_embed(const uint8_t* table, int type, const int* tok, float* x, int d) {
    akd::embed_q_row(table, type, tok + blockIdx.x, x + static_cast<size_t>(blockIdx.x) * d, d);
}

// grid T, 1024 threads. Columns are n apart in x / res / hout / xq / o16; n / 32 apart in xd.
__global__ void __launch_bounds__(1024) k_norm(float* x, const float* res, const float* w, int8_t* xq, float* xd,
                                               __half* o16, int n, float eps, float* hout) {
    const size_t c = blockIdx.x;
    norm_quant(x + c * n, res ? res + c * n : nullptr, w, o16 ? nullptr : xq + c * n, o16 ? nullptr : xd + c * n / 32,
               o16 ? o16 + c * n : nullptr, n, eps, hout ? hout + c * n : nullptr);
}

// MTP input: [RMSNorm(e) * enorm | RMSNorm(h) * hnorm] -> Q8_1 / fp16 with column stride 2d. h columns d apart.
__global__ void __launch_bounds__(1024) k_mtp_in(float* e, float* h, const float* enorm, const float* hnorm,
                                                 int8_t* xq, float* xd, __half* o16, int d, float eps) {
    const size_t c = blockIdx.x;
    if (o16) {
        norm_quant(e + c * d, nullptr, enorm, nullptr, nullptr, o16 + c * 2 * d, d, eps, nullptr);
        norm_quant(h + c * d, nullptr, hnorm, nullptr, nullptr, o16 + c * 2 * d + d, d, eps, nullptr);
    } else {
        norm_quant(e + c * d, nullptr, enorm, xq + c * 2 * d, xd + c * 2 * d / 32, nullptr, d, eps, nullptr);
        norm_quant(h + c * d, nullptr, hnorm, xq + c * 2 * d + d, xd + (c * 2 * d + d) / 32, nullptr, d, eps,
                   nullptr);
    }
}

// Rows one warp of k_gemv_mc covers: K-quant verify passes of 5-12 columns use 4 (two per half-warp, sharing
// activation loads), everything else qgemv_rows_per_warp's 2 (K-quants) or 1 (Q8_0).
// Rows per half-warp in k_gemv_mc (qgemv_dp4a_rows_mc2 shares one activation load across them; 0 = the one-row
// kernel). Measured per width (ms per target pass, docs/QWEN38_9B.md "Wide verify"): <= 4 columns are bandwidth-bound
// and gain nothing; 2 rows win at 5-10 columns; 3 rows at 11-12 (12: 33.5 -> 30.4 ms) but lose at 10 (27.2 -> 29.0).
__host__ __device__ constexpr int mc_r(int type, int NC) {
    return type == 0 || NC <= 4 ? 0 : (NC <= 10 ? 2 : 3);
}
__host__ __device__ constexpr int mc_rows_per_warp(int type, int NC) {
    return mc_r(type, NC) > 0 ? 2 * mc_r(type, NC) : akd::qgemv_rows_per_warp(type);
}

// Multi-column dp4a GEMV, 128 threads per block. Optional second matrix W2: rows >= rows1 come from it (the
// one-row kernel needs rows1 to be a multiple of its rows per block; mm2 checks that).
// Up to 4 columns a pass is bandwidth-bound and 8 blocks / SM (64 registers) hide DRAM latency; wider variants
// (verify passes of 5-12 tokens) carry one accumulator per column and ran out of registers at 64, so they get
// 4 blocks / SM (128 registers): 12 columns 50.3 -> 44.9 ms per target pass. 2 blocks (255) was worse.
template <int type, int NC>
__global__ void __launch_bounds__(128, (NC > 4 ? 4 : 8)) k_gemv_mc(const uint8_t* W, const uint8_t* W2, int rows1, const int8_t* xq,
                                                    const float* xd, float* y, int ystride, int rows, int K) {
    constexpr int bb = type == 0 ? 34 : (type == 1 ? 144 : 210);
    const int nb = type == 0 ? (K >> 5) : (K >> 8);
    const int per_block = (blockDim.x >> 5) * mc_rows_per_warp(type, NC);
    const int row0 = blockIdx.x * per_block;
    if (row0 >= rows) return;
    const int nrows = min(per_block, rows - row0);
    const uint8_t* base = W;
    int wrow = row0;
    if (W2 && row0 >= rows1) {
        base = W2;
        wrow = row0 - rows1;
    }
    if constexpr (mc_r(type, NC) > 0) {
        akd::qgemv_dp4a_rows_mc2<type, NC, mc_r(type, NC)>(W, W2, W2 ? rows1 : rows, xq, xd, y, ystride, row0,
                                                           row0 + nrows, K);
    } else {
        akd::qgemv_dp4a_rows_mc<type, NC>(base + static_cast<size_t>(wrow) * nb * bb, xq, xd, y + row0, ystride,
                                          nrows, K);
    }
}

// Causal depthwise conv over the last 4 inputs of every channel, then SiLU, for T consecutive tokens.
// state is [3][C], oldest first; w is [C][4] (ggml ssm_conv layout, w[ch][3] multiplies the newest input).
// First replays ctl.commit saved inputs of the previous pass (buffer ctl.par). save_mode: the state is left after
// the replay and this pass's inputs go to buffer par ^ 1; otherwise every column runs into the state.
// SNAP: the pass takes recurrent-state snapshots (SnapReq). A template flag rather than a runtime check, because a
// check in k_gdn's inner loop cost ~9% of that kernel on every pass (measured), and snapshot passes are rare.
template <bool SNAP>
__global__ void k_conv(const float* in, int in_stride, float* state, float* save, const int* ctl, const float* w,
                       float* out, int C, int T, int save_mode, SnapReq sr, size_t sr_off) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C) return;
    float s0 = state[ch], s1 = state[C + ch], s2 = state[2 * C + ch];
    const int par = ctl[kCtlPar];
    const int nc = ctl[kCtlCommit];
    const float* sv = save + static_cast<size_t>(par) * kMaxCols * C;
    for (int i = 0; i < nc; ++i) {
        s0 = s1;
        s1 = s2;
        s2 = sv[static_cast<size_t>(i) * C + ch];
    }
    if (save_mode && nc > 0) {
        state[ch] = s0;
        state[C + ch] = s1;
        state[2 * C + ch] = s2;
    }
    float* dst = save + static_cast<size_t>(par ^ 1) * kMaxCols * C;
    const float4 ww = reinterpret_cast<const float4*>(w)[ch];
    for (int c = 0; c < T; ++c) {
        const float xn = in[static_cast<size_t>(c) * in_stride + ch];
        float y = s0 * ww.x;
        y += s1 * ww.y;
        y += s2 * ww.z;
        y += xn * ww.w;
        out[static_cast<size_t>(c) * C + ch] = y * sigmoidf(y);
        if (save_mode) dst[static_cast<size_t>(c) * C + ch] = xn;
        s0 = s1;
        s1 = s2;
        s2 = xn;
        for (int i = 0; SNAP && i < sr.n; ++i) {
            if (sr.col[i] != c) continue;
            float* dst_s = sr.base[i] + sr_off;
            dst_s[ch] = s0;
            dst_s[C + ch] = s1;
            dst_s[2 * C + ch] = s2;
        }
    }
    if (!save_mode) {
        state[ch] = s0;
        state[C + ch] = s1;
        state[2 * C + ch] = s2;
    }
}

// 8-lane sum (the lanes of one state row in k_gdn): xor 4, 2, 1 stays inside the aligned 8-lane group.
__device__ __forceinline__ float sum8(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// One delta-rule step on a state row held by 8 lanes (16 key dims each, as 4 float4 at dims 32 q + 4 sub):
// S *= decay; S += k (v - S.k) beta.
__device__ __forceinline__ void gdn_step8(float4* m, const float4* kk, float decay, float beta, float vj) {
    float sk = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        m[q].x *= decay;
        m[q].y *= decay;
        m[q].z *= decay;
        m[q].w *= decay;
        sk += m[q].x * kk[q].x + m[q].y * kk[q].y + m[q].z * kk[q].z + m[q].w * kk[q].w;
    }
    const float delta = (vj - sum8(sk)) * beta;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        m[q].x += kk[q].x * delta;
        m[q].y += kk[q].y * delta;
        m[q].z += kk[q].z * delta;
        m[q].w += kk[q].w * delta;
    }
}

__device__ __forceinline__ float dot8(const float4* m, const float4* qq) {
    float o = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) o += m[q].x * qq[q].x + m[q].y * qq[q].y + m[q].z * qq[q].z + m[q].w * qq[q].w;
    return sum8(o);
}

// Gated delta rule over T tokens. grid (Hv, kDk / 32), 256 threads. Each warp owns 4 state rows j (value dims) of
// head h for the whole pass, held in registers: 8 lanes per row, lane sub holding key dims 32 q + 4 sub .. +3 for
// q = 0..3 (that interleave keeps the 8 distinct float4 reads of a key / query vector on 8 different bank groups).
// So a row's two 128-dim reductions per token are 3 shuffle steps over 8 lanes, done for the warp's 4 rows at once:
// 6 shuffles per warp and token, against 40 with one row per 32 lanes (2 rows per warp: 20), and shuffles issue at
// one warp-instruction per clock per SM. Value head h reads key/query head h % Hk (ggml layout). conv columns (C
// apart): [q Hk*128 | k Hk*128 | v Hv*128]; ab columns (2 Hv apart): [alpha | beta]. Columns are staged kSub at a
// time. Replay / save as in k_conv.
constexpr int kGdnRows = 32;  // state rows per block (8 warps x 4)
template <bool SNAP>  // as k_conv
__global__ void __launch_bounds__(256) k_gdn(const float* conv, int C, const float* ab, const float* dt_bias,
                                             const float* ssm_a, float* state, float* k_save, float* v_save,
                                             float* db_save, const int* ctl, float* out, int out_stride, int Hk,
                                             int Hv, float eps, int T, int save_mode, SnapReq sr, size_t sr_off) {
    __shared__ __align__(16) float qs[kSub][kDk];
    __shared__ __align__(16) float ks[kSub][kDk];
    __shared__ float vs[kSub][kGdnRows];
    __shared__ float sdecay[kSub], sbeta[kSub];
    const int h = blockIdx.x;
    const int kh = h % Hk;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 3;    // row of the warp's four
    const int sub = lane & 7;   // lane within the row
    const int rl = warp * 4 + g;  // row within the block
    const int j = blockIdx.y * kGdnRows + rl;
    const int d0 = sub * 4;     // first key dim of float4 q is d0 + 32 q
    const size_t col_n = static_cast<size_t>(Hv) * kDk;  // per-column stride of k_save / v_save
    float* Srow = state + (static_cast<size_t>(h) * kDk + j) * kDk;
    float4 m[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) m[q] = *reinterpret_cast<const float4*>(Srow + d0 + 32 * q);
    auto load4 = [&](const float* v, float4* dst) {
#pragma unroll
        for (int q = 0; q < 4; ++q) dst[q] = *reinterpret_cast<const float4*>(v + d0 + 32 * q);
    };
    const int par = ctl[kCtlPar];
    const int nc = ctl[kCtlCommit];
    {
        const float* kp = k_save + static_cast<size_t>(par) * kMaxCols * col_n;
        const float* vp = v_save + static_cast<size_t>(par) * kMaxCols * col_n;
        const float* dp = db_save + static_cast<size_t>(par) * kMaxCols * 2 * Hv;
        for (int i = 0; i < nc; ++i) {
            float4 kk[4];
            load4(kp + i * col_n + h * kDk, kk);
            gdn_step8(m, kk, dp[i * 2 * Hv + h], dp[i * 2 * Hv + Hv + h], vp[i * col_n + h * kDk + j]);
        }
        if (save_mode && nc > 0) {
#pragma unroll
            for (int q = 0; q < 4; ++q) *reinterpret_cast<float4*>(Srow + d0 + 32 * q) = m[q];
        }
    }
    float* kd = k_save + static_cast<size_t>(par ^ 1) * kMaxCols * col_n;
    float* vd = v_save + static_cast<size_t>(par ^ 1) * kMaxCols * col_n;
    float* dd = db_save + static_cast<size_t>(par ^ 1) * kMaxCols * 2 * Hv;
    const float scale = rsqrtf(static_cast<float>(kDk));
    const int jb = blockIdx.y * kGdnRows;
    static_assert(kSub == 8, "k_gdn stages one column per warp (256 threads)");
    for (int c0 = 0; c0 < T; c0 += kSub) {
        const int n = min(kSub, T - c0);
        __syncthreads();  // the previous group's shared values are no longer read
        // Staging: warp cc normalizes column c0 + cc on its own (no block-wide barrier per column). Lane l holds
        // dims l + 32 i; the four 32-lane partial sums are added in a fixed order.
        if (warp < n) {
            const int cc = warp;
            const int c = c0 + cc;
            const float* col = conv + static_cast<size_t>(c) * C;
            float rq[4], rk[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                rq[i] = col[kh * kDk + lane + 32 * i];
                rk[i] = col[Hk * kDk + kh * kDk + lane + 32 * i];
            }
            float tq = 0.f, tk = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                tq += xor_sum(rq[i] * rq[i]);
                tk += xor_sum(rk[i] * rk[i]);
            }
            // L2 norm as ggml builds it: rms_norm(x, eps / n) / sqrt(n) == x / sqrt(sum x^2 + eps).
            const float iq = rsqrtf(tq + eps), ik = rsqrtf(tk + eps);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float nk = rk[i] * ik;
                qs[cc][lane + 32 * i] = rq[i] * iq;
                ks[cc][lane + 32 * i] = nk;
                if (save_mode && blockIdx.y == 0) kd[c * col_n + h * kDk + lane + 32 * i] = nk;
            }
            const float vv = col[2 * Hk * kDk + h * kDk + jb + lane];  // kGdnRows == 32 values, one per lane
            vs[cc][lane] = vv;
            if (save_mode) vd[c * col_n + h * kDk + jb + lane] = vv;
            if (lane == 0) {
                const float* abc = ab + static_cast<size_t>(c) * 2 * Hv;
                const float a = abc[h] + dt_bias[h];
                const float sp = a > 20.f ? a : logf(1.f + expf(a));
                const float decay = expf(sp * ssm_a[h]);
                const float beta = sigmoidf(abc[Hv + h]);
                sdecay[cc] = decay;
                sbeta[cc] = beta;
                if (save_mode && blockIdx.y == 0) {
                    dd[c * 2 * Hv + h] = decay;
                    dd[c * 2 * Hv + Hv + h] = beta;
                }
            }
        }
        __syncthreads();
        for (int cc = 0; cc < n; ++cc) {
            float4 kk[4], qq[4];
            load4(ks[cc], kk);
            load4(qs[cc], qq);
            gdn_step8(m, kk, sdecay[cc], sbeta[cc], vs[cc][rl]);
            const float oj = dot8(m, qq);
            if (sub == 0) out[static_cast<size_t>(c0 + cc) * out_stride + h * kDk + j] = oj * scale;
            for (int i = 0; SNAP && i < sr.n; ++i) {
                if (sr.col[i] != c0 + cc) continue;
                float* dst = sr.base[i] + sr_off + (static_cast<size_t>(h) * kDk + j) * kDk;
#pragma unroll
                for (int q = 0; q < 4; ++q) *reinterpret_cast<float4*>(dst + d0 + 32 * q) = m[q];
            }
        }
    }
    if (!save_mode) {
#pragma unroll
        for (int q = 0; q < 4; ++q) *reinterpret_cast<float4*>(Srow + d0 + 32 * q) = m[q];
    }
}

// Per head: RMSNorm(o) * w * SiLU(z) -> Q8_1 / fp16 for the out projection. grid (Hv, T), 128 threads.
// o and xq / o16 columns are n apart (n = Hv * 128); z columns are z_stride apart.
__global__ void __launch_bounds__(128) k_gated_norm(const float* o, const float* z, int z_stride, const float* w,
                                                    int8_t* xq, float* xd, __half* o16, int n, float eps) {
    __shared__ float red[4];
    const int h = blockIdx.x;
    const size_t c = blockIdx.y;
    const int t = threadIdx.x;
    const float val = o[c * n + h * kDk + t];
    const float ss = xor_sum(val * val);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    const float tot = red[0] + red[1] + red[2] + red[3];
    const float zz = z[c * z_stride + h * kDk + t];
    const float y = val * rsqrtf(tot / kDk + eps) * w[t] * (zz * sigmoidf(zz));
    quant32(y, o16 ? nullptr : xq + c * n, o16 ? nullptr : xd + c * n / 32, o16 ? o16 + c * n : nullptr, h * kDk + t);
}

// q/k RMSNorm per head, NeoX RoPE on the first n_rot dims, k and v appended to the fp16 cache at *pos + c.
// One warp per head (Hq query heads then Hkv key heads), lanes hold dims lane + 32 j. grid (heads / 8, T).
// qk columns (qk_stride apart): [Hq x (q 256 | gate 256) | Hkv x k 256]. q is written compact to qout [c][Hq][256].
__global__ void __launch_bounds__(256) k_attn_prep(const float* qk, int qk_stride, const float* vin, const float* qw,
                                                   const float* kw, float* qout, __half* kc, __half* vc,
                                                   const int* pos_p, int Hq, int Hkv, float eps, float theta,
                                                   int n_rot, int max_seq) {
    const int lane = threadIdx.x & 31;
    const int head = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int c = blockIdx.y;
    if (head >= Hq + Hkv) return;
    const bool isk = head >= Hq;
    const int hh = isk ? head - Hq : head;
    const float* col = qk + static_cast<size_t>(c) * qk_stride;
    const float* src = isk ? col + Hq * 2 * kHd + hh * kHd : col + hh * 2 * kHd;
    const float* w = isk ? kw : qw;
    float vals[8];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        vals[j] = src[lane + 32 * j];
        ss += vals[j] * vals[j];
    }
    ss = xor_sum(ss);
    const float inv = rsqrtf(ss / kHd + eps);
#pragma unroll
    for (int j = 0; j < 8; ++j) vals[j] = vals[j] * inv * w[lane + 32 * j];
    const int pos = *pos_p + c;
    const int half_rot = n_rot / 2;  // pair (i, i + 32) for i < 32 sits in vals[0], vals[1] of lane i
    if (lane < half_rot) {
        const float freq = powf(theta, -2.f * static_cast<float>(lane) / static_cast<float>(n_rot));
        float s, cs;
        sincosf(static_cast<float>(pos) * freq, &s, &cs);
        const float x1 = vals[0], x2 = vals[1];
        vals[0] = x1 * cs - x2 * s;
        vals[1] = x2 * cs + x1 * s;
    }
    if (!isk) {
        float* dst = qout + (static_cast<size_t>(c) * Hq + hh) * kHd;
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[lane + 32 * j] = vals[j];
    } else if (pos < max_seq) {
        const size_t off = (static_cast<size_t>(pos) * Hkv + hh) * kHd;
        const float* vsrc = vin + (static_cast<size_t>(c) * Hkv + hh) * kHd;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            kc[off + lane + 32 * j] = __float2half(vals[j]);
            vc[off + lane + 32 * j] = __float2half(vsrc[lane + 32 * j]);
        }
    }
}

// Split-key attention for one KV head, one slice of kAttnChunk keys and one column, all NR query heads of the group.
// Column c sees keys [0, *pos + c]. Lanes hold 8 contiguous dims (one 16-byte fp16 load per key).
// part[((c * Hq + h) * n_split + s) * (hd + 2)] = {max, sum, acc[hd]}. grid (Hkv, n_split, T), 256 threads.
template <int NR>
__global__ void __launch_bounds__(256) k_attn_part(const float* q, const __half* kc, const __half* vc, float* part,
                                                   const int* pos_p, int Hkv, int n_split, int max_seq) {
    __shared__ float sm_m[8][NR];
    __shared__ float sm_l[8][NR];
    __shared__ float sm_o[8][NR][kHd];
    const int g = blockIdx.x;
    const int s = blockIdx.y;
    const int c = blockIdx.z;
    const int Hq = Hkv * NR;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int seq = min(*pos_p + c + 1, max_seq);
    const int t0 = s * kAttnChunk;
    if (t0 >= seq) return;
    const int t1 = min(seq, t0 + kAttnChunk);
    const float scale = rsqrtf(static_cast<float>(kHd));
    float qr[NR][8], acc[NR][8], m[NR], l[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        m[r] = -1.0e30f;
        l[r] = 0.f;
        const float* qh = q + (static_cast<size_t>(c) * Hq + g * NR + r) * kHd + lane * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            qr[r][j] = qh[j] * scale;
            acc[r][j] = 0.f;
        }
    }
    for (int t = t0 + warp; t < t1; t += 8) {
        const size_t off = (static_cast<size_t>(t) * Hkv + g) * kHd + lane * 8;
        const uint4 kraw = *reinterpret_cast<const uint4*>(kc + off);
        const uint4 vraw = *reinterpret_cast<const uint4*>(vc + off);
        const __half2* k2 = reinterpret_cast<const __half2*>(&kraw);
        const __half2* v2 = reinterpret_cast<const __half2*>(&vraw);
        float kf[8], vf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(k2[j]);
            const float2 b = __half22float2(v2[j]);
            kf[2 * j] = a.x;
            kf[2 * j + 1] = a.y;
            vf[2 * j] = b.x;
            vf[2 * j + 1] = b.y;
        }
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            float dot = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) dot += qr[r][j] * kf[j];
            dot = xor_sum(dot);
            const float mn = fmaxf(m[r], dot);
            const float corr = expf(m[r] - mn);
            const float p = expf(dot - mn);
            l[r] = l[r] * corr + p;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[r][j] = acc[r][j] * corr + p * vf[j];
            m[r] = mn;
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        if (lane == 0) {
            sm_m[warp][r] = m[r];
            sm_l[warp][r] = l[r];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) sm_o[warp][r][lane * 8 + j] = acc[r][j];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < NR * kHd; idx += blockDim.x) {
        const int r = idx / kHd;
        const int dd = idx - r * kHd;
        float M = -1.0e30f;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, sm_m[w][r]);
        float L = 0.f, o = 0.f;
        for (int w = 0; w < 8; ++w) {
            const float e = sm_l[w][r] > 0.f ? expf(sm_m[w][r] - M) : 0.f;
            L += sm_l[w][r] * e;
            o += sm_o[w][r][dd] * e;
        }
        float* dst = part + ((static_cast<size_t>(c) * Hq + g * NR + r) * n_split + s) * (kHd + 2);
        if (dd == 0) {
            dst[0] = M;
            dst[1] = L;
        }
        dst[2 + dd] = o;
    }
}

// Merge the used slices of query head h for column c, multiply by sigmoid(gate), write Q8_1 for the o projection
// (columns Hq * 256 apart). grid (Hq, T), 256 threads (one per dim).
__global__ void __launch_bounds__(256) k_attn_combine(const float* part, const float* qk, int qk_stride, int8_t* oq,
                                                      float* od, const int* pos_p, int n_split, int max_seq) {
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    const int Hq = gridDim.x;
    const int dd = threadIdx.x;
    const int seq = min(*pos_p + c + 1, max_seq);
    const int used = min(n_split, (seq + kAttnChunk - 1) / kAttnChunk);
    const float* base = part + (static_cast<size_t>(c) * Hq + h) * n_split * (kHd + 2);
    float M = -1.0e30f;
    for (int s = 0; s < used; ++s) M = fmaxf(M, base[s * (kHd + 2)]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < used; ++s) {
        const float* p = base + s * (kHd + 2);
        const float e = p[1] > 0.f ? expf(p[0] - M) : 0.f;
        L += p[1] * e;
        o += p[2 + dd] * e;
    }
    float val = L > 0.f ? o / L : 0.f;
    val *= sigmoidf(qk[static_cast<size_t>(c) * qk_stride + h * 2 * kHd + kHd + dd]);
    const size_t n = static_cast<size_t>(Hq) * kHd;
    quant32(val, oq + c * n, od + c * n / 32, nullptr, h * kHd + dd);
}

// Prefill attention: one block per (KV head, column), all keys [0, *pos + c] in one pass (8 warps over interleaved
// keys, merged in shared memory), then * sigmoid(gate) -> fp16 o-projection input (columns Hq * 256 apart).
template <int NR>
__global__ void __launch_bounds__(256) k_attn_pf(const float* q, const __half* kc, const __half* vc, const float* qk,
                                                 int qk_stride, __half* o16, const int* pos_p, int Hkv, int max_seq) {
    __shared__ float sm_m[8][NR];
    __shared__ float sm_l[8][NR];
    __shared__ float sm_o[8][NR][kHd];
    const int g = blockIdx.x;
    const int c = blockIdx.y;
    const int Hq = Hkv * NR;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int seq = min(*pos_p + c + 1, max_seq);
    const float scale = rsqrtf(static_cast<float>(kHd));
    float qr[NR][8], acc[NR][8], m[NR], l[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        m[r] = -1.0e30f;
        l[r] = 0.f;
        const float* qh = q + (static_cast<size_t>(c) * Hq + g * NR + r) * kHd + lane * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            qr[r][j] = qh[j] * scale;
            acc[r][j] = 0.f;
        }
    }
    for (int t = warp; t < seq; t += 8) {
        const size_t off = (static_cast<size_t>(t) * Hkv + g) * kHd + lane * 8;
        const uint4 kraw = *reinterpret_cast<const uint4*>(kc + off);
        const uint4 vraw = *reinterpret_cast<const uint4*>(vc + off);
        const __half2* k2 = reinterpret_cast<const __half2*>(&kraw);
        const __half2* v2 = reinterpret_cast<const __half2*>(&vraw);
        float kf[8], vf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(k2[j]);
            const float2 b = __half22float2(v2[j]);
            kf[2 * j] = a.x;
            kf[2 * j + 1] = a.y;
            vf[2 * j] = b.x;
            vf[2 * j + 1] = b.y;
        }
        float dots[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            float dot = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) dot += qr[r][j] * kf[j];
            dots[r] = dot;
        }
        // NR butterfly reductions interleaved, so their shuffles overlap.
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (int r = 0; r < NR; ++r) dots[r] += __shfl_xor_sync(0xffffffff, dots[r], o);
        }
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float mn = fmaxf(m[r], dots[r]);
            const float corr = expf(m[r] - mn);
            const float p = expf(dots[r] - mn);
            l[r] = l[r] * corr + p;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[r][j] = acc[r][j] * corr + p * vf[j];
            m[r] = mn;
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        if (lane == 0) {
            sm_m[warp][r] = m[r];
            sm_l[warp][r] = l[r];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) sm_o[warp][r][lane * 8 + j] = acc[r][j];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < NR * kHd; idx += blockDim.x) {
        const int r = idx / kHd;
        const int dd = idx - r * kHd;
        float M = -1.0e30f;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, sm_m[w][r]);
        float L = 0.f, o = 0.f;
        for (int w = 0; w < 8; ++w) {
            const float e = sm_l[w][r] > 0.f ? expf(sm_m[w][r] - M) : 0.f;
            L += sm_l[w][r] * e;
            o += sm_o[w][r][dd] * e;
        }
        const int h = g * NR + r;
        float val = L > 0.f ? o / L : 0.f;
        val *= sigmoidf(qk[static_cast<size_t>(c) * qk_stride + h * 2 * kHd + kHd + dd]);
        o16[static_cast<size_t>(c) * Hq * kHd + h * kHd + dd] = __float2half(val);
    }
}

// ------------------------------------------------------------------------------------------------
// Tensor-core attention (FlashAttention-style). A block holds 64 query rows - (column, head) pairs of one KV head's
// group, row R = c * NR + head-in-group - in 4 warps of 16, and streams keys in tiles of 32 through shared memory,
// so every key is read once per block instead of once per column. S = Q K^T and O += P V run as
// mma.m16n8k16 (fp16 in, fp32 accumulate) with an online softmax; P stays in registers (the S accumulator layout of
// two 8-key tiles is the A-operand layout of one 16-key step).
//   !SPLIT (prefill): all keys [0, pos + c], finalize, * sigmoid(gate) -> fp16 o-projection input.
//   SPLIT (decode / verify): keys of chunk blockIdx.y (kAttnChunk keys) -> unnormalized partials for k_attn_combine.
// ------------------------------------------------------------------------------------------------

constexpr int kAttnBK = 32;           // keys per shared-memory tile
constexpr int kAttnRows = 64;         // query rows per block (4 warps x 16)
constexpr int kAttnLd = kHd + 8;      // padded row stride (halves) against bank conflicts
constexpr size_t kAttnSmem = static_cast<size_t>(kAttnRows + 2 * kAttnBK) * kAttnLd * sizeof(__half);
static_assert(kHd == 256, "k_attn_mma assumes head dim 256 (scale 1/16)");
static_assert(kAttnChunk % kAttnBK == 0, "split chunk must be whole key tiles");

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t ld_h2(const __half* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ uint32_t pack_f2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_h2(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

template <int NR, bool SPLIT>
__global__ void __launch_bounds__(128) k_attn_mma(const float* q, const __half* kc, const __half* vc, const float* qk,
                                                  int qk_stride, __half* o16, float* part, const int* pos_p, int Hkv,
                                                  int T, int n_split, int max_seq) {
    extern __shared__ __align__(16) unsigned char attn_smem[];
    __half* Qs = reinterpret_cast<__half*>(attn_smem);
    __half* Ks = Qs + kAttnRows * kAttnLd;
    __half* Vs = Ks + kAttnBK * kAttnLd;
    const int g = blockIdx.x;
    const int split = SPLIT ? static_cast<int>(blockIdx.y) : 0;
    const int tile = SPLIT ? blockIdx.z : blockIdx.y;
    const int Hq = Hkv * NR;
    const int pos = *pos_p;
    const int n_rows = T * NR;
    const int row0 = tile * kAttnRows;
    const int last_row = min(row0 + kAttnRows, n_rows) - 1;
    const int seq_max = min(pos + last_row / NR + 1, max_seq);  // the block needs keys [0, seq_max)
    int k_begin = 0, k_end = seq_max;
    if (SPLIT) {
        k_begin = split * kAttnChunk;
        k_end = min(seq_max, k_begin + kAttnChunk);
    }
    if (k_begin >= k_end) return;  // uniform across the block

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gq = lane >> 2;
    const int tig = lane & 3;

    // Q tile -> fp16, pre-scaled by 1/sqrt(256). Padding rows are zero.
    for (int i = threadIdx.x; i < kAttnRows * (kHd / 2); i += blockDim.x) {
        const int r = i / (kHd / 2);
        const int d2 = (i - r * (kHd / 2)) * 2;
        const int R = row0 + r;
        float2 v = make_float2(0.f, 0.f);
        if (R < n_rows) {
            const int c = R / NR;
            const int h = g * NR + (R - c * NR);
            v = *reinterpret_cast<const float2*>(q + (static_cast<size_t>(c) * Hq + h) * kHd + d2);
        }
        *reinterpret_cast<__half2*>(Qs + r * kAttnLd + d2) = __floats2half2_rn(v.x * 0.0625f, v.y * 0.0625f);
    }

    // The two rows this thread accumulates and the last key each may see (-1: padding row).
    const int rA = row0 + warp * 16 + gq;
    const int rB = rA + 8;
    const int limA = rA < n_rows ? min(pos + rA / NR, max_seq - 1) : -1;
    const int limB = rB < n_rows ? min(pos + rB / NR, max_seq - 1) : -1;

    float o[kHd / 8][4];
#pragma unroll
    for (int nt = 0; nt < kHd / 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    float mA = -1.0e30f, mB = -1.0e30f, lA = 0.f, lB = 0.f;

    for (int kt = k_begin; kt < k_end; kt += kAttnBK) {
        __syncthreads();  // previous tile consumed (first pass: Q stores visible)
        for (int i = threadIdx.x; i < 2 * kAttnBK * (kHd / 8); i += blockDim.x) {
            const int mat = i / (kAttnBK * (kHd / 8));
            const int j = i - mat * (kAttnBK * (kHd / 8));
            const int kr = j / (kHd / 8);
            const int d8 = (j - kr * (kHd / 8)) * 8;
            const int t = kt + kr;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < k_end) v = *reinterpret_cast<const uint4*>((mat ? vc : kc) + (static_cast<size_t>(t) * Hkv + g) * kHd + d8);
            *reinterpret_cast<uint4*>((mat ? Vs : Ks) + kr * kAttnLd + d8) = v;
        }
        __syncthreads();

        // S = Q K^T for this warp's 16 rows x 32 keys (4 n-tiles of 8 keys).
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
        const __half* qa = Qs + (warp * 16 + gq) * kAttnLd + tig * 2;
#pragma unroll
        for (int ks = 0; ks < kHd / 16; ++ks) {
            uint32_t a[4];
            a[0] = ld_h2(qa + ks * 16);
            a[1] = ld_h2(qa + 8 * kAttnLd + ks * 16);
            a[2] = ld_h2(qa + ks * 16 + 8);
            a[3] = ld_h2(qa + 8 * kAttnLd + ks * 16 + 8);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const __half* kb = Ks + (nt * 8 + gq) * kAttnLd + ks * 16 + tig * 2;
                uint32_t b[2] = {ld_h2(kb), ld_h2(kb + 8)};
                mma16816(s[nt], a, b);
            }
        }

        // Online softmax. Thread holds row A at s[nt][0..1] and row B at s[nt][2..3]; a row spans a lane quad.
        float mxA = -1.0e30f, mxB = -1.0e30f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int key = kt + nt * 8 + tig * 2 + j;
                if (key > limA) s[nt][j] = -1.0e30f;
                if (key > limB) s[nt][2 + j] = -1.0e30f;
                mxA = fmaxf(mxA, s[nt][j]);
                mxB = fmaxf(mxB, s[nt][2 + j]);
            }
        }
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, 1));
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, 2));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, 1));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, 2));
        const float nA = fmaxf(mA, mxA), nB = fmaxf(mB, mxB);
        const float alA = __expf(mA - nA), alB = __expf(mB - nB);
        mA = nA;
        mB = nB;
        float psA = 0.f, psB = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int key = kt + nt * 8 + tig * 2 + j;
                s[nt][j] = key <= limA ? __expf(s[nt][j] - nA) : 0.f;
                s[nt][2 + j] = key <= limB ? __expf(s[nt][2 + j] - nB) : 0.f;
                psA += s[nt][j];
                psB += s[nt][2 + j];
            }
        }
        lA = lA * alA + psA;
        lB = lB * alB + psB;
#pragma unroll
        for (int nt = 0; nt < kHd / 8; ++nt) {
            o[nt][0] *= alA;
            o[nt][1] *= alA;
            o[nt][2] *= alB;
            o[nt][3] *= alB;
        }

        // O += P V, two 16-key steps over 32 dim tiles of 8.
#pragma unroll
        for (int kc2 = 0; kc2 < 2; ++kc2) {
            uint32_t a[4];
            a[0] = pack_f2(s[2 * kc2][0], s[2 * kc2][1]);
            a[1] = pack_f2(s[2 * kc2][2], s[2 * kc2][3]);
            a[2] = pack_f2(s[2 * kc2 + 1][0], s[2 * kc2 + 1][1]);
            a[3] = pack_f2(s[2 * kc2 + 1][2], s[2 * kc2 + 1][3]);
            const __half* vb = Vs + (kc2 * 16 + tig * 2) * kAttnLd + gq;
#pragma unroll
            for (int nt = 0; nt < kHd / 8; ++nt) {
                const __half* v0 = vb + nt * 8;
                uint32_t b[2];
                b[0] = pack_h2(v0[0], v0[kAttnLd]);
                b[1] = pack_h2(v0[8 * kAttnLd], v0[9 * kAttnLd]);
                mma16816(o[nt], a, b);
            }
        }
    }

    lA += __shfl_xor_sync(0xffffffff, lA, 1);
    lA += __shfl_xor_sync(0xffffffff, lA, 2);
    lB += __shfl_xor_sync(0xffffffff, lB, 1);
    lB += __shfl_xor_sync(0xffffffff, lB, 2);

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int R = half ? rB : rA;
        if (R >= n_rows) continue;
        const int c = R / NR;
        const int h = g * NR + (R - c * NR);
        const float l = half ? lB : lA;
        if (SPLIT) {
            float* dst = part + ((static_cast<size_t>(c) * Hq + h) * n_split + split) * (kHd + 2);
            if (tig == 0) {
                dst[0] = half ? mB : mA;
                dst[1] = l;
            }
#pragma unroll
            for (int nt = 0; nt < kHd / 8; ++nt) {
                const int d = nt * 8 + tig * 2;
                dst[2 + d] = o[nt][2 * half];
                dst[3 + d] = o[nt][2 * half + 1];
            }
        } else {
            const float inv = l > 0.f ? 1.f / l : 0.f;
            const float* gate = qk + static_cast<size_t>(c) * qk_stride + h * 2 * kHd + kHd;
            __half* dst = o16 + static_cast<size_t>(c) * Hq * kHd + h * kHd;
#pragma unroll
            for (int nt = 0; nt < kHd / 8; ++nt) {
                const int d = nt * 8 + tig * 2;
                const float v0 = o[nt][2 * half] * inv * sigmoidf(gate[d]);
                const float v1 = o[nt][2 * half + 1] * inv * sigmoidf(gate[d + 1]);
                *reinterpret_cast<__half2*>(dst + d) = __floats2half2_rn(v0, v1);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Tensor-core attention, v2 (`k_attn_mma2`, the default). Same tiling, masking and outputs as k_attn_mma; two changes
// in how data reaches the tensor cores (docs/QWEN38_9B.md, "Attention v2"):
//   1. Every mma operand comes from one ldmatrix instead of 32-bit / 16-bit shared loads. v1 issued 12 LDS.32 per
//      four S mmas and 4 LDS.16 + 2 packs per PV mma, so the PV half was load-issue bound; ldmatrix.x4 gives the
//      Q fragment of one 16-dim step, the K fragments of two 8-key tiles, or (with .trans, since V is stored key-major
//      and PV needs it dim-major) the V fragments of two 8-dim tiles.
//   2. K and V tiles stream in with cp.async, overlapped with compute: V(t) is copied while S = Q K(t)^T and the
//      softmax run, and K(t+1) while O += P V(t) runs. Splitting the pipeline by operand rather than double-buffering
//      both keeps shared memory at v1's 66 KB (a full K+V double buffer would need the whole 99 KB).
// ------------------------------------------------------------------------------------------------

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const __half* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s)
                 : "memory");
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const __half* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s)
                 : "memory");
}

__device__ __forceinline__ void attn_cp16(__half* smem, const __half* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void attn_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void attn_cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// One 32-key tile of K or V (rows kt..kt+31 of KV head g) -> shared, asynchronously. Rows >= k_end are zero-filled
// (src-size 0), so masked keys never multiply garbage (0 * NaN) in P V.
__device__ __forceinline__ void attn_tile_async(__half* dst, const __half* src, int kt, int k_end, int Hkv, int g) {
#pragma unroll
    for (int j = 0; j < kAttnBK * (kHd / 8) / 128; ++j) {
        const int i = threadIdx.x + j * 128;
        const int kr = i / (kHd / 8);
        const int d8 = (i - kr * (kHd / 8)) * 8;
        const int t = kt + kr;
        const bool ok = t < k_end;
        attn_cp16(dst + kr * kAttnLd + d8, src + (static_cast<size_t>(ok ? t : 0) * Hkv + g) * kHd + d8, ok);
    }
}

template <int NR, bool SPLIT>
__global__ void __launch_bounds__(128) k_attn_mma2(const float* q, const __half* kc, const __half* vc, const float* qk,
                                                   int qk_stride, __half* o16, float* part, const int* pos_p, int Hkv,
                                                   int T, int n_split, int max_seq) {
    extern __shared__ __align__(16) unsigned char attn_smem[];
    __half* Qs = reinterpret_cast<__half*>(attn_smem);
    __half* Ks = Qs + kAttnRows * kAttnLd;
    __half* Vs = Ks + kAttnBK * kAttnLd;
    const int g = blockIdx.x;
    const int split = SPLIT ? static_cast<int>(blockIdx.y) : 0;
    const int tile = SPLIT ? blockIdx.z : blockIdx.y;
    const int Hq = Hkv * NR;
    const int pos = *pos_p;
    const int n_rows = T * NR;
    const int row0 = tile * kAttnRows;
    const int last_row = min(row0 + kAttnRows, n_rows) - 1;
    const int seq_max = min(pos + last_row / NR + 1, max_seq);
    int k_begin = 0, k_end = seq_max;
    if (SPLIT) {
        k_begin = split * kAttnChunk;
        k_end = min(seq_max, k_begin + kAttnChunk);
    }
    if (k_begin >= k_end) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gq = lane >> 2;
    const int tig = lane & 3;

    attn_tile_async(Ks, kc, k_begin, k_end, Hkv, g);  // first K tile in flight while Q is converted
    attn_cp_commit();
    for (int i = threadIdx.x; i < kAttnRows * (kHd / 2); i += blockDim.x) {
        const int r = i / (kHd / 2);
        const int d2 = (i - r * (kHd / 2)) * 2;
        const int R = row0 + r;
        float2 v = make_float2(0.f, 0.f);
        if (R < n_rows) {
            const int c = R / NR;
            const int h = g * NR + (R - c * NR);
            v = *reinterpret_cast<const float2*>(q + (static_cast<size_t>(c) * Hq + h) * kHd + d2);
        }
        *reinterpret_cast<__half2*>(Qs + r * kAttnLd + d2) = __floats2half2_rn(v.x * 0.0625f, v.y * 0.0625f);
    }

    const int rA = row0 + warp * 16 + gq;
    const int rB = rA + 8;
    const int limA = rA < n_rows ? min(pos + rA / NR, max_seq - 1) : -1;
    const int limB = rB < n_rows ? min(pos + rB / NR, max_seq - 1) : -1;

    // Per-lane ldmatrix row addresses (lane l supplies the address of one 8-element row of one of the 4 matrices).
    //   Q (A, 16 rows x 16 dims): matrices (rows 0-7 | 8-15) x (dims 0-7 | 8-15) -> a[0..3].
    //   K (B of S, non-trans): keys (l & 7) + 8 * (l >> 4), dims 8 * ((l >> 3) & 1) -> b[nt][0..1], b[nt+1][0..1].
    //   V (B of PV, .trans): keys (l & 7) + 8 * ((l >> 3) & 1), dims 8 * (l >> 4) -> b[nt][0..1], b[nt+1][0..1].
    const __half* q_ld = Qs + (warp * 16 + (lane & 15)) * kAttnLd + (lane >> 4) * 8;
    const __half* k_ld = Ks + ((lane & 7) + 8 * (lane >> 4)) * kAttnLd + 8 * ((lane >> 3) & 1);
    const __half* v_ld = Vs + ((lane & 7) + 8 * ((lane >> 3) & 1)) * kAttnLd + 8 * (lane >> 4);

    float o[kHd / 8][4];
#pragma unroll
    for (int nt = 0; nt < kHd / 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    float mA = -1.0e30f, mB = -1.0e30f, lA = 0.f, lB = 0.f;

    for (int kt = k_begin; kt < k_end; kt += kAttnBK) {
        attn_cp_wait_all();
        __syncthreads();  // K(kt) and Q visible; every warp is done with V(kt - BK)
        attn_tile_async(Vs, vc, kt, k_end, Hkv, g);
        attn_cp_commit();

        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < kHd / 16; ++ks) {
            uint32_t a[4], kb[2][4];
            ldsm_x4(a, q_ld + ks * 16);
            ldsm_x4(kb[0], k_ld + ks * 16);
            ldsm_x4(kb[1], k_ld + 16 * kAttnLd + ks * 16);
            mma16816(s[0], a, &kb[0][0]);
            mma16816(s[1], a, &kb[0][2]);
            mma16816(s[2], a, &kb[1][0]);
            mma16816(s[3], a, &kb[1][2]);
        }

        float mxA = -1.0e30f, mxB = -1.0e30f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int key = kt + nt * 8 + tig * 2 + j;
                if (key > limA) s[nt][j] = -1.0e30f;
                if (key > limB) s[nt][2 + j] = -1.0e30f;
                mxA = fmaxf(mxA, s[nt][j]);
                mxB = fmaxf(mxB, s[nt][2 + j]);
            }
        }
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, 1));
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, 2));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, 1));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, 2));
        const float nA = fmaxf(mA, mxA), nB = fmaxf(mB, mxB);
        const float alA = __expf(mA - nA), alB = __expf(mB - nB);
        mA = nA;
        mB = nB;
        float psA = 0.f, psB = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int key = kt + nt * 8 + tig * 2 + j;
                s[nt][j] = key <= limA ? __expf(s[nt][j] - nA) : 0.f;
                s[nt][2 + j] = key <= limB ? __expf(s[nt][2 + j] - nB) : 0.f;
                psA += s[nt][j];
                psB += s[nt][2 + j];
            }
        }
        lA = lA * alA + psA;
        lB = lB * alB + psB;
#pragma unroll
        for (int nt = 0; nt < kHd / 8; ++nt) {
            o[nt][0] *= alA;
            o[nt][1] *= alA;
            o[nt][2] *= alB;
            o[nt][3] *= alB;
        }

        attn_cp_wait_all();
        __syncthreads();  // V(kt) visible; every warp is done with K(kt)
        if (kt + kAttnBK < k_end) attn_tile_async(Ks, kc, kt + kAttnBK, k_end, Hkv, g);
        attn_cp_commit();

#pragma unroll
        for (int kc2 = 0; kc2 < 2; ++kc2) {
            uint32_t a[4];
            a[0] = pack_f2(s[2 * kc2][0], s[2 * kc2][1]);
            a[1] = pack_f2(s[2 * kc2][2], s[2 * kc2][3]);
            a[2] = pack_f2(s[2 * kc2 + 1][0], s[2 * kc2 + 1][1]);
            a[3] = pack_f2(s[2 * kc2 + 1][2], s[2 * kc2 + 1][3]);
            const __half* vb = v_ld + kc2 * 16 * kAttnLd;
#pragma unroll
            for (int nt = 0; nt < kHd / 8; nt += 2) {
                uint32_t b[4];
                ldsm_x4_t(b, vb + nt * 8);
                mma16816(o[nt], a, &b[0]);
                mma16816(o[nt + 1], a, &b[2]);
            }
        }
    }

    lA += __shfl_xor_sync(0xffffffff, lA, 1);
    lA += __shfl_xor_sync(0xffffffff, lA, 2);
    lB += __shfl_xor_sync(0xffffffff, lB, 1);
    lB += __shfl_xor_sync(0xffffffff, lB, 2);

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int R = half ? rB : rA;
        if (R >= n_rows) continue;
        const int c = R / NR;
        const int h = g * NR + (R - c * NR);
        const float l = half ? lB : lA;
        if (SPLIT) {
            float* dst = part + ((static_cast<size_t>(c) * Hq + h) * n_split + split) * (kHd + 2);
            if (tig == 0) {
                dst[0] = half ? mB : mA;
                dst[1] = l;
            }
#pragma unroll
            for (int nt = 0; nt < kHd / 8; ++nt) {
                const int d = nt * 8 + tig * 2;
                dst[2 + d] = o[nt][2 * half];
                dst[3 + d] = o[nt][2 * half + 1];
            }
        } else {
            const float inv = l > 0.f ? 1.f / l : 0.f;
            const float* gate = qk + static_cast<size_t>(c) * qk_stride + h * 2 * kHd + kHd;
            __half* dst = o16 + static_cast<size_t>(c) * Hq * kHd + h * kHd;
#pragma unroll
            for (int nt = 0; nt < kHd / 8; ++nt) {
                const int d = nt * 8 + tig * 2;
                const float v0 = o[nt][2 * half] * inv * sigmoidf(gate[d]);
                const float v1 = o[nt][2 * half + 1] * inv * sigmoidf(gate[d + 1]);
                *reinterpret_cast<__half2*>(dst + d) = __floats2half2_rn(v0, v1);
            }
        }
    }
}

// silu(gate) * up -> Q8_1 / fp16, gu columns [gate ff | up ff]. grid (ff / 256, T).
__global__ void k_silu_quant(const float* gu, int8_t* xq, float* xd, __half* o16, int ff) {
    const size_t c = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float g = gu[c * 2 * ff + i];
    quant32(g * sigmoidf(g) * gu[c * 2 * ff + ff + i], o16 ? nullptr : xq + c * ff, o16 ? nullptr : xd + c * ff / 32,
            o16 ? o16 + c * ff : nullptr, i);
}

// grid (ceil(n / 4096), T): fold logits column c into best[c].
__global__ void k_argmax(const float* logits, int n, unsigned long long* best) {
    const int i0 = blockIdx.x * 4096;
    akd::argmax_part(logits + static_cast<size_t>(blockIdx.y) * n + i0, min(4096, n - i0), i0, best + blockIdx.y);
}

// out[c] = argmax index of column c, then clear best for the next pass.
__global__ void k_argmax_final(unsigned long long* best, int* out, int T) {
    const int c = threadIdx.x;
    if (c < T) {
        out[c] = akd::argmax_key_index(best[c]);
        best[c] = 0ull;
    }
}

// Plain decode chaining: append arg[0] to out, feed it back, advance one position, commit the one column.
__global__ void k_plain_next(const int* arg, int* tok, int* ctl, int* out, int* count) {
    if (threadIdx.x == 0) {
        const int id = arg[0];
        out[*count] = id;
        *count += 1;
        tok[0] = id;
        ctl[kCtlPos] += 1;
        ctl[kCtlCommit] = 1;
        ctl[kCtlPar] ^= 1;
    }
}

__global__ void k_advance(int* p, int n) {
    if (threadIdx.x == 0) *p += n;
}

// ------------------------------------------------------------------------------------------------
// Passes
// ------------------------------------------------------------------------------------------------

struct Prof {
    std::vector<cudaEvent_t> ev;
    std::vector<int> kind;
};

void mark(Prof* p, cudaStream_t s, int kind) {
    if (!p) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, s);
    p->ev.push_back(e);
    p->kind.push_back(kind);
}

template <int type>
void launch_mc(int T, const uint8_t* W, const uint8_t* W2, int rows1, const int8_t* xq, const float* xd, float* y,
               int ystride, int rows, int K, cudaStream_t s) {
    const int per_block = 4 * mc_rows_per_warp(type, std::min(T, kMaxCols));
    const int blocks = (rows + per_block - 1) / per_block;
#define AK_Q35_MC(N) \
    case N: k_gemv_mc<type, N><<<blocks, 128, 0, s>>>(W, W2, rows1, xq, xd, y, ystride, rows, K); break;
    switch (T) {
        AK_Q35_MC(2)
        AK_Q35_MC(3)
        AK_Q35_MC(4)
        AK_Q35_MC(5)
        AK_Q35_MC(6)
        AK_Q35_MC(7)
        AK_Q35_MC(8)
        AK_Q35_MC(9)
        AK_Q35_MC(10)
        AK_Q35_MC(11)
        default: k_gemv_mc<type, 12><<<blocks, 128, 0, s>>>(W, W2, rows1, xq, xd, y, ystride, rows, K); break;
    }
#undef AK_Q35_MC
}

// y[c * ystride + row] = A (or [A; B]) x column c. Activation columns are A.K apart (xq) and A.K / 32 (xd).
ak_status gemv(const QMat& A, const QMat* B, const int8_t* xq, const float* xd, float* y, int ystride, int T,
               cudaStream_t s) {
    const int rows = A.rows + (B ? B->rows : 0);
    if (T == 1) {
        ak_task t{};
        t.op = A.type == 0 ? AK_OP_Q8_DP4A : (A.type == 1 ? AK_OP_Q4K_DP4A : AK_OP_Q6K_DP4A);
        t.p0 = reinterpret_cast<uint64_t>(A.w);
        t.p1 = reinterpret_cast<uint64_t>(xq);
        t.p2 = reinterpret_cast<uint64_t>(y);
        t.p3 = reinterpret_cast<uint64_t>(xd);
        t.n0 = rows;
        t.n1 = A.K;
        if (B) {
            t.p4 = reinterpret_cast<uint64_t>(B->w);
            t.n2 = A.rows;
        }
        return ak_launch_task(&t, s);
    }
    const uint8_t* W2 = B ? B->w : nullptr;
    if (A.type == 0) launch_mc<0>(T, A.w, W2, A.rows, xq, xd, y, ystride, rows, A.K, s);
    else if (A.type == 1) launch_mc<1>(T, A.w, W2, A.rows, xq, xd, y, ystride, rows, A.K, s);
    else launch_mc<2>(T, A.w, W2, A.rows, xq, xd, y, ystride, rows, A.K, s);
    return ck(cudaGetLastError());
}

// Prefill matrix product: Y[c][0..rows) (row stride ldy) = W . X[c], X fp16 [T][W.K]. W is expanded to fp16 into
// m->w16, then cuBLAS (tensor cores, fp32 accumulate).
// Prefill matrix product for one chunk. Up to qgemm_max columns the fused dequant + mma kernel (qgemm.cuh) reads the
// 4/6-bit weights directly; longer chunks expand the matrix to fp16 once and use cuBLAS.
ak_status pf_gemm(ak_q35_model* m, const QMat& W, const __half* X, int T, float* Y, int ldy) {
    cudaStream_t s = m->stream;
    if (T <= m->qgemm_max && (W.type == 1 || W.type == 2) && W.K % 256 == 0) {
        using namespace akd::qgemm;
        auto launch = [&](auto kern, int bn) {
            kern<<<dim3((T + bn - 1) / bn, (W.rows + BM - 1) / BM), NT, 0, s>>>(W.w, X, Y, W.rows, W.K, T, ldy);
        };
        if (T <= 128) {
            if (W.type == 1) launch(k_qgemm<1, 64>, 64);
            else launch(k_qgemm<2, 64>, 64);
        } else {
            if (W.type == 1) launch(k_qgemm<1, 128>, 128);
            else launch(k_qgemm<2, 128>, 128);
        }
        return ck(cudaGetLastError());
    }
    const size_t n = static_cast<size_t>(W.rows) * W.K;
    if (!m->w16) {  // only chunks longer than qgemm_max need it (~100 MB for this model)
        const ak_status st = dalloc_n(m, &m->w16, m->w16_n);
        if (st != AK_OK) return st;
        m->info.vram_mb = static_cast<float>(m->dev_bytes / (1024.0 * 1024.0));
    }
    if (n > m->w16_n) {
        ak_set_error("qwen35 prefill: fp16 scratch too small");
        return AK_ERR_INVALID;
    }
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((n / 2 + threads - 1) / threads, 65535u * 8u));
    if (W.type == 0) akd::pf_dequant_f16<0><<<blocks, threads, 0, s>>>(W.w, m->w16, W.rows, W.K);
    else if (W.type == 1) akd::pf_dequant_f16<1><<<blocks, threads, 0, s>>>(W.w, m->w16, W.rows, W.K);
    else akd::pf_dequant_f16<2><<<blocks, threads, 0, s>>>(W.w, m->w16, W.rows, W.K);
    const float alpha = 1.f, beta = 0.f;
    const cublasStatus_t cs = cublasGemmEx(m->blas, CUBLAS_OP_T, CUBLAS_OP_N, W.rows, T, W.K, &alpha, m->w16,
                                           CUDA_R_16F, W.K, X, CUDA_R_16F, W.K, &beta, Y, CUDA_R_32F, ldy,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        ak_set_error("qwen35 prefill: cublasGemmEx failed");
        return AK_ERR_CUDA;
    }
    return ck(cudaGetLastError());
}

// Where a producer kernel writes the activations a matrix product reads: Q8_1 (decode) or fp16 (prefill).
struct Act {
    int8_t* q;
    float* d;
    __half* h;
};
Act act_hq(const Bufs& b) { return b.pf ? Act{nullptr, nullptr, b.a16} : Act{b.hq_q, b.hq_d, nullptr}; }
Act act_aq(const Bufs& b) { return b.pf ? Act{nullptr, nullptr, b.a16} : Act{b.aq, b.ad, nullptr}; }
Act act_fq(const Bufs& b) { return b.pf ? Act{nullptr, nullptr, b.a16} : Act{b.fq, b.fd, nullptr}; }

// Pipelined int8 GEMM for one Q4_K / Q6_K matrix. BN tokens per block, S cp.async stages, MB blocks per SM; the
// shared-memory limit is raised on first use of each instantiation.
template <int BN, int S, int MB>
void mmq_launch(const QMat& W, const int8_t* xq, const float* xd, const float* xsd, float* y, int T, int ldy,
                int nsplit, cudaStream_t s) {
    using namespace akd::qgemm;
    const dim3 g((T + BN - 1) / BN, (W.rows + BM - 1) / BM, nsplit);
    if (W.type == 1) {
        static bool init = false;
        if (!init) {
            cudaFuncSetAttribute(reinterpret_cast<const void*>(k_mmq4<BN, S, MB>),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(p4_smem_bytes<BN, S>()));
            init = true;
        }
        k_mmq4<BN, S, MB><<<g, NT, p4_smem_bytes<BN, S>(), s>>>(W.w, xq, xd, xsd, y, W.rows, W.K, T, ldy, nsplit);
    } else {
        static bool init = false;
        if (!init) {
            cudaFuncSetAttribute(reinterpret_cast<const void*>(k_mmq6<BN, S, MB>),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(p6_smem_bytes<BN, S>()));
            init = true;
        }
        k_mmq6<BN, S, MB><<<g, NT, p6_smem_bytes<BN, S>(), s>>>(W.w, xq, xd, xsd, y, W.rows, W.K, T, ldy, nsplit);
    }
}

// Configurations (AK_Q35_MMQ_SHORT for chunks <= 64 tokens, default 0; AK_Q35_MMQ_LONG above, default 2):
//   0: BN 64, 3 stages, 1 block / SM     1: BN 64, 2 stages, 2 blocks / SM     2: BN 128, 2 stages, 1 block / SM
// RTX 4060 GEMM ms per chunk (T = 64 / 128 / 256 / 512 / 1024):
//   0: 49.1 / 77.6 / 149.7 / 288.6 / 560.7   1: 54.8 / 83.9 / 155.0 / 296.4 / 572.4   2: 70.2 / 66.1 / 119.1 / 234.1 / 440.7
// Tile size (data reuse) matters more than occupancy.
void mmq_dispatch(const QMat& W, const int8_t* xq, const float* xd, const float* xsd, float* y, int T, int ldy,
                  int cfg, int nsplit, cudaStream_t s) {
    if (cfg == 1) mmq_launch<64, 2, 2>(W, xq, xd, xsd, y, T, ldy, nsplit, s);
    else if (cfg == 2) mmq_launch<128, 2, 1>(W, xq, xd, xsd, y, T, ldy, nsplit, s);
    else mmq_launch<64, 3, 1>(W, xq, xd, xsd, y, T, ldy, nsplit, s);
}

// K splits for one int8 GEMM launch. Blocks run one per SM, so a grid of `blocks` takes ceil(blocks / n_sm) waves;
// with 4096-row matrices and short chunks that is 1.33 or 2.67 waves, and the last wave leaves most SMs idle.
// Splitting K into z parts makes z times as many blocks, each 1/z as long: time ~ ceil(z * blocks / slots) / z.
// Pick the z in 1..max that minimizes it (ties: fewer splits), with z dividing the super-blocks of K, at least 4
// super-blocks (1024 K) per split, the partial sums fitting the scratch buffer, and a predicted gain of >= 20%.
int splitk_choose(const ak_q35_model* m, int blocks, int per_sm, int K, int T, int rows) {
    if (m->splitk_max <= 1 || m->n_sm <= 0) return 1;
    const int sb = K / 256;
    const int slots = m->n_sm * per_sm;  // blocks resident at once
    int best = 1;
    const double t1 = std::ceil(static_cast<double>(blocks) / slots);
    double best_t = t1;
    for (int z = 2; z <= m->splitk_max; ++z) {
        if (sb % z != 0 || sb / z < 4) continue;
        if (static_cast<size_t>(z) * T * rows > m->splitk_n) continue;
        const double t = std::ceil(static_cast<double>(z) * blocks / slots) / z;
        // Measured (benchmarks/gemm_bench.cu): the wave model overstates small gains (the reduction pass and the
        // shorter per-block K loops cost something), so only split when it predicts at least 20% less time.
        if (t < best_t - 1e-9 && t <= 0.8 * t1) {
            best_t = t;
            best = z;
        }
    }
    return best;
}

// y = A (or [A; B]) x for T columns of b's activations.
ak_status mm(ak_q35_model* m, const Bufs& b, const QMat& A, const QMat* B, Act a, float* y, int ystride, int T) {
    if (!b.pf) return gemv(A, B, a.q, a.d, y, ystride, T, m->stream);
    auto int8_ok = [&](const QMat& W) { return (W.type == 1 || W.type == 2) && W.K % 256 == 0; };
    // int8 tensor cores (qgemm.cuh) are the fastest path at every chunk length; fp16 fused / cuBLAS otherwise.
    if (m->int8_gemm && T <= m->int8_max && int8_ok(A) && (!B || (int8_ok(*B) && B->K == A.K))) {
        using namespace akd::qgemm;
        cudaStream_t s = m->stream;
        // quantize the activations once for both matrices
        k_quant_x8<<<dim3(A.K / 256, T), 256, 0, s>>>(a.h, A.K, m->xq8, m->xd8, m->xsd8);
        for (int i = 0; i < (B ? 2 : 1); ++i) {
            const QMat& W = i ? *B : A;
            float* yy = i ? y + A.rows : y;
            auto launch = [&](auto kern, int bn, int bm) {
                kern<<<dim3((T + bn - 1) / bn, (W.rows + bm - 1) / bm), NT, 0, s>>>(W.w, m->xq8, m->xd8, m->xsd8, yy,
                                                                                   W.rows, W.K, T, ystride, m->qdiag);
            };
            if (m->qdiag != 2) {  // pipelined cp.async kernels (AK_Q35_QDIAG=2: the plain register-staged ones)
                const int cfg = T <= 64 ? m->mmq_short : m->mmq_long;
                const int bn = cfg == 2 ? 128 : 64;
                const int blocks = ((T + bn - 1) / bn) * ((W.rows + BM - 1) / BM);
                const int z = splitk_choose(m, blocks, cfg == 1 ? 2 : 1, W.K, T, W.rows);
                if (z == 1) {
                    mmq_dispatch(W, m->xq8, m->xd8, m->xsd8, yy, T, ystride, cfg, 1, s);
                } else {
                    mmq_dispatch(W, m->xq8, m->xd8, m->xsd8, m->splitk, T, W.rows, cfg, z, s);
                    k_splitk_sum<<<dim3((W.rows + 255) / 256, T), 256, 0, s>>>(m->splitk, z, W.rows, T, yy, ystride);
                }
            } else if (T <= 128) {
                if (W.type == 1) launch(k_qgemm8<1, 128, 64>, 64, 128);
                else launch(k_qgemm8<2, 128, 64>, 64, 128);
            } else {
                if (W.type == 1) launch(k_qgemm8<1, 128, 128>, 128, 128);
                else launch(k_qgemm8<2, 128, 128>, 128, 128);
            }
        }
        return ck(cudaGetLastError());
    }
    ak_status st = pf_gemm(m, A, a.h, T, y, ystride);
    if (st == AK_OK && B) st = pf_gemm(m, *B, a.h, T, y + A.rows, ystride);
    return st;
}

// Two matrices with the same type and K go out as one GEMV launch writing [A rows | B rows].
ak_status mm2(ak_q35_model* m, const Bufs& b, const QMat& A, const QMat& B, Act a, float* y, int ystride, int T) {
    // one launch for both: a block must not straddle A and B (16 rows per block in wide verify passes)
    if (b.pf || (A.type == B.type && A.K == B.K && (A.rows % 16) == 0)) return mm(m, b, A, &B, a, y, ystride, T);
    ak_status st = mm(m, b, A, nullptr, a, y, ystride, T);
    if (st == AK_OK) st = mm(m, b, B, nullptr, a, y + A.rows, ystride, T);
    return st;
}

// Full-attention block on the normed input (act_hq) -> proj. pos_p: position of column 0.
ak_status emit_attn(ak_q35_model* m, const Bufs& b, const Layer& L, int T, const int* pos_p, Prof* p) {
    cudaStream_t s = m->stream;
    float* qk = b.big;  // per column: [Hq * 512 | Hkv * 256]
    const Act hq = act_hq(b);
    ak_status st = mm2(m, b, L.wq, L.wk, hq, qk, m->big_n, T);
    if (st == AK_OK) st = mm(m, b, L.wv, nullptr, hq, b.v, m->hkv * kHd, T);
    mark(p, s, kKGemvAttnIn);
    k_attn_prep<<<dim3((m->hq + m->hkv + 7) / 8, T), 256, 0, s>>>(qk, m->big_n, b.v, L.q_norm, L.k_norm, b.q, L.kc,
                                                                  L.vc, pos_p, m->hq, m->hkv, m->eps, m->theta,
                                                                  m->n_rot, m->max_seq);
    mark(p, s, kKAttnPrep);
    const int ratio = m->hq / m->hkv;
    if (m->attn != 0) {
        const int tiles = (T * ratio + kAttnRows - 1) / kAttnRows;
        auto launch = [&](auto kern, dim3 grid) {
            kern<<<grid, 128, kAttnSmem, s>>>(b.q, L.kc, L.vc, qk, m->big_n, b.a16, m->part, pos_p, m->hkv, T,
                                              m->n_split, m->max_seq);
        };
        const bool v2 = m->attn == 2;
        if (b.pf) {
            const dim3 grid(m->hkv, tiles);
            if (ratio == 1) launch(v2 ? k_attn_mma2<1, false> : k_attn_mma<1, false>, grid);
            else if (ratio == 2) launch(v2 ? k_attn_mma2<2, false> : k_attn_mma<2, false>, grid);
            else launch(v2 ? k_attn_mma2<4, false> : k_attn_mma<4, false>, grid);
            mark(p, s, kKAttnPart);
        } else {
            const dim3 grid(m->hkv, m->n_split, tiles);
            if (ratio == 1) launch(v2 ? k_attn_mma2<1, true> : k_attn_mma<1, true>, grid);
            else if (ratio == 2) launch(v2 ? k_attn_mma2<2, true> : k_attn_mma<2, true>, grid);
            else launch(v2 ? k_attn_mma2<4, true> : k_attn_mma<4, true>, grid);
            mark(p, s, kKAttnPart);
            k_attn_combine<<<dim3(m->hq, T), kHd, 0, s>>>(m->part, qk, m->big_n, b.aq, b.ad, pos_p, m->n_split,
                                                          m->max_seq);
            mark(p, s, kKAttnCombine);
        }
    } else if (b.pf) {
        const dim3 grid(m->hkv, T);
        if (ratio == 1) k_attn_pf<1><<<grid, 256, 0, s>>>(b.q, L.kc, L.vc, qk, m->big_n, b.a16, pos_p, m->hkv, m->max_seq);
        else if (ratio == 2) k_attn_pf<2><<<grid, 256, 0, s>>>(b.q, L.kc, L.vc, qk, m->big_n, b.a16, pos_p, m->hkv, m->max_seq);
        else k_attn_pf<4><<<grid, 256, 0, s>>>(b.q, L.kc, L.vc, qk, m->big_n, b.a16, pos_p, m->hkv, m->max_seq);
        mark(p, s, kKAttnPart);
    } else {
        const dim3 grid(m->hkv, m->n_split, T);
        if (ratio == 1) k_attn_part<1><<<grid, 256, 0, s>>>(b.q, L.kc, L.vc, m->part, pos_p, m->hkv, m->n_split, m->max_seq);
        else if (ratio == 2) k_attn_part<2><<<grid, 256, 0, s>>>(b.q, L.kc, L.vc, m->part, pos_p, m->hkv, m->n_split, m->max_seq);
        else k_attn_part<4><<<grid, 256, 0, s>>>(b.q, L.kc, L.vc, m->part, pos_p, m->hkv, m->n_split, m->max_seq);
        mark(p, s, kKAttnPart);
        k_attn_combine<<<dim3(m->hq, T), kHd, 0, s>>>(m->part, qk, m->big_n, b.aq, b.ad, pos_p, m->n_split, m->max_seq);
        mark(p, s, kKAttnCombine);
    }
    if (st == AK_OK) st = mm(m, b, L.wo, nullptr, act_aq(b), b.proj, m->d, T);
    mark(p, s, kKGemvAttnOut);
    return st;
}

// post-attention norm (adds proj into x) and SwiGLU FFN -> proj.
ak_status emit_ffn(ak_q35_model* m, const Bufs& b, const Layer& L, int T, Prof* p) {
    cudaStream_t s = m->stream;
    const Act hq = act_hq(b), fq = act_fq(b);
    k_norm<<<T, 1024, 0, s>>>(b.x, b.proj, L.post_norm, hq.q, hq.d, hq.h, m->d, m->eps, nullptr);
    mark(p, s, kKNorm);
    ak_status st = mm2(m, b, L.wgate, L.wup, hq, b.gu, 2 * m->ff, T);
    mark(p, s, kKGemvGateUp);
    k_silu_quant<<<dim3(m->ff / 256, T), 256, 0, s>>>(b.gu, fq.q, fq.d, fq.h, m->ff);
    mark(p, s, kKSilu);
    if (st == AK_OK) st = mm(m, b, L.wdown, nullptr, fq, b.proj, m->d, T);
    mark(p, s, kKGemvDown);
    return st;
}

// argmax of the vocab logits of T columns -> out[0..T).
void emit_argmax(ak_q35_model* m, const float* logits, int rows, int T, unsigned long long* best, int* out) {
    k_argmax<<<dim3((rows + 4095) / 4096, T), 256, 0, m->stream>>>(logits, rows, best);
    k_argmax_final<<<1, 32, 0, m->stream>>>(best, out, T);
}

// Target pass over T columns of b: tokens b.tok[0..T), positions ctl.pos + c. save_mode: decode / verify (replay
// ctl.commit, save this pass's inputs); otherwise prefill (columns run into the state).
// Decode buffers: normed final hidden of every column -> hbuf, logits and greedy token of every column -> d_arg.
// Prefill buffers: normed hidden of every column -> b.hbuf; only the last column gets logits, into m->logits column
// 0, its hidden into dec.hbuf column 0 and its greedy token into d_arg[0].
ak_status emit_target(ak_q35_model* m, const Bufs& b, int T, int save_mode, Prof* p) {
    cudaStream_t s = m->stream;
    const int d = m->d;
    const int* pos_p = m->d_ctl + kCtlPos;
    const int* ctl = m->d_ctl;
    const Act hq = act_hq(b), aq = act_aq(b);
    ak_status st = AK_OK;
    k_embed<<<T, 256, 0, s>>>(m->embd_dev, m->embd_type, b.tok, b.x, d);
    mark(p, s, kKEmbed);
    const float* pending = nullptr;
    const SnapReq sr = m->snapreq;
    const size_t snap_nc = 3 * static_cast<size_t>(m->conv_ch);
    const size_t snap_layer = snap_nc + static_cast<size_t>(m->hv) * kDk * kDk;
    size_t li = 0;  // linear layers so far (index into a snapshot slot)
    for (int l = 0; l < m->n_layers && st == AK_OK; ++l) {
        const Layer& L = m->layers[l];
        k_norm<<<T, 1024, 0, s>>>(b.x, pending, L.attn_norm, hq.q, hq.d, hq.h, d, m->eps, nullptr);
        mark(p, s, kKNorm);
        if (L.linear) {
            const size_t sr_off = li++ * snap_layer;
            float* qkv = b.big;  // per column: [conv_ch | z d_inner]
            st = mm2(m, b, L.wqkv, L.wz, hq, qkv, m->big_n, T);
            mark(p, s, kKGemvQkv);
            if (st == AK_OK) st = mm2(m, b, L.walpha, L.wbeta, hq, b.ab, 2 * m->hv, T);
            mark(p, s, kKGemvAb);
            (sr.n ? k_conv<true> : k_conv<false>)<<<(m->conv_ch + 255) / 256, 256, 0, s>>>(
                qkv, m->big_n, L.conv_state, L.conv_save, ctl, L.conv_w, b.conv_out, m->conv_ch, T, save_mode, sr,
                sr_off);
            mark(p, s, kKConv);
            (sr.n ? k_gdn<true> : k_gdn<false>)<<<dim3(m->hv, kDk / kGdnRows), 256, 0, s>>>(
                b.conv_out, m->conv_ch, b.ab, L.dt_bias, L.ssm_a, L.gdn_state, L.k_save, L.v_save, L.db_save, ctl,
                b.o, m->d_inner, m->hk, m->hv, m->eps, T, save_mode, sr, sr_off + snap_nc);
            mark(p, s, kKGdn);
            k_gated_norm<<<dim3(m->hv, T), kDk, 0, s>>>(b.o, qkv + m->conv_ch, m->big_n, L.ssm_norm, aq.q, aq.d, aq.h,
                                                        m->d_inner, m->eps);
            mark(p, s, kKGatedNorm);
            if (st == AK_OK) st = mm(m, b, L.wout, nullptr, aq, b.proj, d, T);
            mark(p, s, kKGemvOut);
        } else {
            st = emit_attn(m, b, L, T, pos_p, p);
        }
        if (st == AK_OK) st = emit_ffn(m, b, L, T, p);
        pending = b.proj;
    }
    if (!b.pf) {
        k_norm<<<T, 1024, 0, s>>>(b.x, pending, m->out_norm, b.hq_q, b.hq_d, nullptr, d, m->eps, b.hbuf);
        mark(p, s, kKNorm);
        if (st == AK_OK) st = gemv(m->out, nullptr, b.hq_q, b.hq_d, m->logits, m->vocab, T, s);
        mark(p, s, kKGemvLogits);
        emit_argmax(m, m->logits, m->vocab, T, m->best, m->d_arg);
        mark(p, s, kKArgmax);
    } else {
        k_norm<<<T, 1024, 0, s>>>(b.x, pending, m->out_norm, nullptr, nullptr, b.a16, d, m->eps, b.hbuf);
        const Bufs& D = m->dec;
        k_norm<<<1, 1024, 0, s>>>(b.x + static_cast<size_t>(T - 1) * d, nullptr, m->out_norm, D.hq_q, D.hq_d, nullptr,
                                  d, m->eps, D.hbuf);
        if (st == AK_OK) st = gemv(m->out, nullptr, D.hq_q, D.hq_d, m->logits, m->vocab, 1, s);
        emit_argmax(m, m->logits, m->vocab, 1, m->best, m->d_arg);
    }
    if (st == AK_OK) st = ck(cudaGetLastError());
    return st;
}

// MTP pass over T columns of b: pairs (h[c], tok[c]) at MTP positions ctl.mpos + c; advances ctl.mpos by T. With
// out_tok (decode buffers only), the last column goes through the draft LM head (the first draft_vocab rows of the
// output matrix) and its greedy token is written to *out_tok; the normed output hidden of every column goes to mh
// (input of the next chained draft).
ak_status emit_mtp(ak_q35_model* m, const Bufs& b, int T, const float* h, const int* tok, int* out_tok) {
    cudaStream_t s = m->stream;
    const int d = m->d;
    const Mtp& M = m->mtp;
    int* mpos = m->d_ctl + kCtlMpos;
    const Act hq = act_hq(b);
    k_embed<<<T, 256, 0, s>>>(m->embd_dev, m->embd_type, tok, b.ebuf, d);
    // h is read from outside (hbuf / mh); copy it so k_mtp_in may run on scratch.
    ak_status st = ck(cudaMemcpyAsync(b.x, h, sizeof(float) * d * T, cudaMemcpyDeviceToDevice, s));
    k_mtp_in<<<T, 1024, 0, s>>>(b.ebuf, b.x, M.enorm, M.hnorm, hq.q, hq.d, hq.h, d, m->eps);
    if (st == AK_OK) st = mm(m, b, M.eh_proj, nullptr, hq, b.x, d, T);
    k_norm<<<T, 1024, 0, s>>>(b.x, nullptr, M.blk.attn_norm, hq.q, hq.d, hq.h, d, m->eps, nullptr);
    if (st == AK_OK) st = emit_attn(m, b, M.blk, T, mpos, nullptr);
    if (st == AK_OK) st = emit_ffn(m, b, M.blk, T, nullptr);
    if (!b.pf) {
        k_norm<<<T, 1024, 0, s>>>(b.x, b.proj, M.head_norm, b.hq_q, b.hq_d, nullptr, d, m->eps, b.mh);
        if (out_tok && st == AK_OK) {
            const size_t lc = static_cast<size_t>(T - 1);
            QMat head = m->out;
            head.rows = m->draft_vocab;
            st = gemv(head, nullptr, b.hq_q + lc * d, b.hq_d + lc * d / 32, m->logits, head.rows, 1, s);
            emit_argmax(m, m->logits, head.rows, 1, m->best + kMaxCols, out_tok);
        }
    }
    k_advance<<<1, 32, 0, s>>>(mpos, T);
    if (st == AK_OK) st = ck(cudaGetLastError());
    return st;
}

// Graph cache. key[0] = kind: 0 target, 1 target + plain chaining, 2 MTP.
template <class F>
ak_status run_graph(ak_q35_model* m, const std::vector<uint64_t>& key, F emit) {
    auto it = m->graphs.find(key);
    if (it == m->graphs.end()) {
        const auto t0 = std::chrono::steady_clock::now();
        ak_status st = ck(cudaStreamBeginCapture(m->stream, cudaStreamCaptureModeGlobal));
        if (st != AK_OK) return st;
        st = emit();
        cudaGraph_t g = nullptr;
        const cudaError_t end = cudaStreamEndCapture(m->stream, &g);
        if (st != AK_OK) {
            cudaGetLastError();
            if (g) cudaGraphDestroy(g);
            return st;
        }
        if (end != cudaSuccess) return ck(end);
        cudaGraphExec_t ex = nullptr;
        st = ck(cudaGraphInstantiate(&ex, g, 0));
        cudaGraphDestroy(g);
        if (st != AK_OK) return st;
        it = m->graphs.emplace(key, ex).first;
        m->capture_ms += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    return ck(cudaGraphLaunch(it->second, m->stream));
}

ak_status target(ak_q35_model* m, int T, int save_mode, int use_graph) {
    if (!use_graph) return emit_target(m, m->dec, T, save_mode, nullptr);
    return run_graph(m, {0, static_cast<uint64_t>(T), static_cast<uint64_t>(save_mode)},
                     [&] { return emit_target(m, m->dec, T, save_mode, nullptr); });
}

ak_status target_plain(ak_q35_model* m, int use_graph) {
    auto emit = [&] {
        ak_status st = emit_target(m, m->dec, 1, 1, nullptr);
        k_plain_next<<<1, 32, 0, m->stream>>>(m->d_arg, m->dec.tok, m->d_ctl, m->d_out, m->d_count);
        return st == AK_OK ? ck(cudaGetLastError()) : st;
    };
    if (!use_graph) return emit();
    return run_graph(m, {1}, emit);
}

ak_status mtp(ak_q35_model* m, int T, const float* h, const int* tok, int* out_tok) {
    const std::vector<uint64_t> key{2, static_cast<uint64_t>(T), reinterpret_cast<uint64_t>(h),
                                    reinterpret_cast<uint64_t>(tok), reinterpret_cast<uint64_t>(out_tok),
                                    static_cast<uint64_t>(m->draft_vocab)};
    return run_graph(m, key, [&] { return emit_mtp(m, m->dec, T, h, tok, out_tok); });
}

int* pin_ctl(ak_q35_model* m) { return m->h_pin; }
int* pin_tok(ak_q35_model* m) { return m->h_pin + kCtlN; }
int* pin_mtok(ak_q35_model* m) { return m->h_pin + kCtlN + kMaxCols; }
int* pin_res(ak_q35_model* m) { return m->h_pin + kCtlN + 2 * kMaxCols; }

// Stage the control block. The pinned buffer is only rewritten after the stream has drained.
ak_status set_ctl(ak_q35_model* m, int pos, int commit, int par, int mpos) {
    int* c = pin_ctl(m);
    c[kCtlPos] = pos;
    c[kCtlCommit] = commit;
    c[kCtlPar] = par;
    c[kCtlMpos] = mpos;
    return ck(cudaMemcpyAsync(m->d_ctl, c, sizeof(int) * kCtlN, cudaMemcpyHostToDevice, m->stream));
}

ak_status reset(ak_q35_model* m) {
    cudaStream_t s = m->stream;
    ak_status st = AK_OK;
    for (Layer& L : m->layers) {
        if (!L.linear) continue;  // KV caches need no clearing: attention reads only positions written this sequence
        if (st == AK_OK) st = ck(cudaMemsetAsync(L.conv_state, 0, sizeof(float) * 3 * m->conv_ch, s));
        if (st == AK_OK) st = ck(cudaMemsetAsync(L.gdn_state, 0, sizeof(float) * m->hv * kDk * kDk, s));
    }
    if (st == AK_OK) st = ck(cudaMemsetAsync(m->d_count, 0, sizeof(int), s));
    if (st == AK_OK) st = ck(cudaMemsetAsync(m->best, 0, sizeof(unsigned long long) * (kMaxCols + 1), s));
    if (st == AK_OK) st = ck(cudaStreamSynchronize(s));
    return st;
}

// Save (or restore) every linear layer's conv + delta-rule state to (from) snapshot slot `slot`. Only valid when
// no replay is pending, i.e. right after a prefill.
// Allocate snapshot slot `slot` on first use.
ak_status snap_ensure(ak_q35_model* m, int slot) {
    if (slot < 0 || slot >= AK_Q35_CACHE_SLOTS) {
        ak_set_error("qwen35 cache: slot out of range");
        return AK_ERR_INVALID;
    }
    if (m->snap[slot]) return AK_OK;
    size_t n_lin = 0;
    for (const Layer& L : m->layers) n_lin += L.linear ? 1 : 0;
    const size_t per = 3 * static_cast<size_t>(m->conv_ch) + static_cast<size_t>(m->hv) * kDk * kDk;
    const ak_status st = dalloc_n(m, &m->snap[slot], n_lin * per);
    m->info.vram_mb = static_cast<float>(m->dev_bytes / (1024.0 * 1024.0));
    return st;
}

ak_status snap_copy(ak_q35_model* m, int slot, bool save) {
    if (slot < 0 || slot >= AK_Q35_CACHE_SLOTS) {
        ak_set_error("qwen35 cache: slot out of range");
        return AK_ERR_INVALID;
    }
    const size_t nc = 3 * static_cast<size_t>(m->conv_ch);
    const size_t ng = static_cast<size_t>(m->hv) * kDk * kDk;
    if (!m->snap[slot] && !save) {
        ak_set_error("qwen35 cache: slot is empty");
        return AK_ERR_INVALID;
    }
    ak_status st = snap_ensure(m, slot);
    if (st != AK_OK) return st;
    float* p = m->snap[slot];
    cudaStream_t s = m->stream;
    for (Layer& L : m->layers) {
        if (!L.linear) continue;
        float* dc = save ? p : L.conv_state;
        const float* sc = save ? L.conv_state : p;
        float* dg = save ? p + nc : L.gdn_state;
        const float* sg = save ? L.gdn_state : p + nc;
        if (st == AK_OK) st = ck(cudaMemcpyAsync(dc, sc, sizeof(float) * nc, cudaMemcpyDeviceToDevice, s));
        if (st == AK_OK) st = ck(cudaMemcpyAsync(dg, sg, sizeof(float) * ng, cudaMemcpyDeviceToDevice, s));
        p += nc + ng;
    }
    if (st == AK_OK) st = ck(cudaStreamSynchronize(s));
    return st;
}

// Activation buffers for `cols` columns. Decode buffers hold Q8_1 activations, prefill buffers fp16 ones.
ak_status alloc_bufs(ak_q35_model* m, Bufs* b, int cols, bool pf) {
    const size_t C = static_cast<size_t>(cols);
    const size_t d = m->d;
    b->cols = cols;
    b->pf = pf;
    ak_status st = dalloc_n(m, &b->x, C * d);
    if (st == AK_OK) st = dalloc_n(m, &b->proj, C * d);
    if (st == AK_OK) st = dalloc_n(m, &b->ebuf, C * d);
    if (st == AK_OK) st = dalloc_n(m, &b->hbuf, C * d);
    if (st == AK_OK && !pf) st = dalloc_n(m, &b->mh, C * d);
    if (st == AK_OK) st = dalloc_n(m, &b->big, C * m->big_n);
    if (st == AK_OK) st = dalloc_n(m, &b->ab, C * 2 * m->hv);
    if (st == AK_OK) st = dalloc_n(m, &b->conv_out, C * m->conv_ch);
    if (st == AK_OK) st = dalloc_n(m, &b->o, C * m->d_inner);
    if (st == AK_OK) st = dalloc_n(m, &b->q, C * m->hq * kHd);
    if (st == AK_OK) st = dalloc_n(m, &b->v, C * m->hkv * kHd);
    if (st == AK_OK) st = dalloc_n(m, &b->gu, C * 2 * m->ff);
    if (st == AK_OK) st = dalloc_n(m, &b->tok, C);
    if (st == AK_OK) st = dalloc_n(m, &b->mtok, C);
    if (pf) {
        const size_t kmax = std::max<size_t>(std::max<size_t>(m->ff, 2 * d), m->aq_n);
        if (st == AK_OK) st = dalloc_n(m, &b->a16, C * kmax);
    } else {
        const size_t hq_n = 2 * d;  // MTP concat input is 2d wide
        if (st == AK_OK) st = dalloc_n(m, &b->hq_q, C * hq_n);
        if (st == AK_OK) st = dalloc_n(m, &b->hq_d, C * hq_n / 32);
        if (st == AK_OK) st = dalloc_n(m, &b->aq, C * m->aq_n);
        if (st == AK_OK) st = dalloc_n(m, &b->ad, C * m->aq_n / 32);
        if (st == AK_OK) st = dalloc_n(m, &b->fq, C * m->ff);
        if (st == AK_OK) st = dalloc_n(m, &b->fd, C * m->ff / 32);
    }
    return st;
}

// Prefill buffers, fp16 scratch for the largest matrix (the LM head is never expanded) and the cuBLAS handle.
ak_status ensure_pf(ak_q35_model* m) {
    if (m->pf.cols) return AK_OK;
    const char* env = std::getenv("AK_Q35_PF_COLS");
    int cols = env ? std::atoi(env) : kPfCols;
    cols = std::max(kPfMin + 1, std::min(cols, 4096));
    size_t wmax = 0;
    auto see = [&](const Layer& L) {
        for (const QMat* q : {&L.wqkv, &L.wz, &L.walpha, &L.wbeta, &L.wout, &L.wq, &L.wk, &L.wv, &L.wo, &L.wgate,
                              &L.wup, &L.wdown}) {
            wmax = std::max(wmax, static_cast<size_t>(q->rows) * q->K);
        }
    };
    for (const Layer& L : m->layers) see(L);
    if (m->mtp.present) {
        see(m->mtp.blk);
        wmax = std::max(wmax, static_cast<size_t>(m->mtp.eh_proj.rows) * m->mtp.eh_proj.K);
    }
    ak_status st = ck(cublasCreate(&m->blas) == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown);
    if (st == AK_OK) {
        cublasSetStream(m->blas, m->stream);
        cublasSetMathMode(m->blas, CUBLAS_DEFAULT_MATH);
        m->w16_n = wmax;  // the fp16 scratch itself is allocated on the first cuBLAS chunk (pf_gemm)
    }
    {
        const char* qg = std::getenv("AK_Q35_QGEMM_MAX");
        if (qg) m->qgemm_max = std::atoi(qg);
        const char* i8 = std::getenv("AK_Q35_INT8");
        if (i8) m->int8_gemm = i8[0] != '0';
        const char* i8m = std::getenv("AK_Q35_INT8_MAX");
        if (i8m) m->int8_max = std::atoi(i8m);
        const char* qd = std::getenv("AK_Q35_QDIAG");
        if (qd) m->qdiag = std::atoi(qd);
        const char* ms = std::getenv("AK_Q35_MMQ_SHORT");
        if (ms) m->mmq_short = std::atoi(ms);
        const char* ml = std::getenv("AK_Q35_MMQ_LONG");
        if (ml) m->mmq_long = std::atoi(ml);
        const char* sk = std::getenv("AK_Q35_SPLITK");
        if (sk) m->splitk_max = std::max(1, std::atoi(sk));
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&m->n_sm, cudaDevAttrMultiProcessorCount, dev);
    }
    {  // int8 activations: the widest K of any prefill matrix (FFN down: ff, MTP eh_proj: 2 d)
        const size_t kmax = std::max<size_t>(m->ff, 2 * static_cast<size_t>(m->d));
        if (st == AK_OK) st = dalloc_n(m, &m->xq8, kmax * cols);
        if (st == AK_OK) st = dalloc_n(m, &m->xd8, kmax / 32 * cols);
        if (st == AK_OK) st = dalloc_n(m, &m->xsd8, kmax / 32 * cols);
        // split-K partials: room for 4 splits of a d-row matrix over 256 tokens (16 MB); larger cases do not split
        m->splitk_n = static_cast<size_t>(4) * 256 * m->d;
        if (st == AK_OK) st = dalloc_n(m, &m->splitk, m->splitk_n);
    }
    if (st == AK_OK) st = ck(cudaHostAlloc(reinterpret_cast<void**>(&m->h_pf), sizeof(int) * 2 * cols, 0));
    Bufs b;
    if (st == AK_OK) st = alloc_bufs(m, &b, cols, true);
    if (st != AK_OK) return st;
    m->pf = b;
    m->pf_cols = cols;
    m->info.vram_mb = static_cast<float>(m->dev_bytes / (1024.0 * 1024.0));
    return AK_OK;
}

// Feed the prompt: chunks of up to pf_cols tokens through the GEMM path while more than kPfMin tokens remain, the
// rest kMaxCols at a time through the GEMV path. Afterwards the greedy next token is d_arg[*last_col], its hidden is
// dec.hbuf column *last_col and its logits m->logits column *last_col. With fill_mtp, the MTP KV cache gets every
// prompt pair (h_i, t_{i+1}).
// Only positions [begin, end) are processed (end < 0: n); MTP pairs still look ahead to prompt[end] when end < n.
// snaps: {position, slot} pairs, increasing, in (begin, end]: the recurrent state after prompt[0, position) is saved
// into the slot. Each is taken inside the pass that covers it (SnapReq), so snapshots do not split the prompt.
// GEMM chunks are balanced (rem tokens -> ceil(rem / pf_cols) equal chunks), so no remainder of <= kPfMin tokens is
// left for the GEMV path, which reads all weights once per kMaxCols tokens.
ak_status prefill(ak_q35_model* m, const int32_t* prompt, int n, int use_graph, bool fill_mtp, int* last_col,
                  int begin = 0, int end = -1, const std::vector<std::pair<int, int>>* snaps = nullptr) {
    ak_status st = AK_OK;
    cudaStream_t s = m->stream;
    if (end < 0) end = n;
    size_t next_snap = 0;
    for (int s0 = begin; s0 < end && st == AK_OK;) {
        const int rem = end - s0;
        int T;
        const bool gemm = rem > kPfMin && !m->no_pf && (st = ensure_pf(m)) == AK_OK;
        if (st != AK_OK) break;
        if (gemm) {
            const int parts = (rem + m->pf_cols - 1) / m->pf_cols;
            T = std::min(m->pf_cols, (rem + parts - 1) / parts);
        } else {
            T = std::min(kMaxCols, rem);
        }
        SnapReq sr{};
        while (snaps && next_snap < snaps->size() && (*snaps)[next_snap].first <= s0 + T) {
            const auto [at, slot] = (*snaps)[next_snap++];
            if (at <= s0) continue;  // not inside this range (caller error); skip rather than save a wrong state
            st = snap_ensure(m, slot);
            if (st != AK_OK) break;
            sr.base[sr.n] = m->snap[slot];
            sr.col[sr.n] = at - s0 - 1;
            ++sr.n;
        }
        if (st != AK_OK) break;
        m->snapreq = sr;
        if (gemm) {
            const Bufs& b = m->pf;
            std::memcpy(m->h_pf, prompt + s0, sizeof(int) * T);
            st = ck(cudaMemcpyAsync(b.tok, m->h_pf, sizeof(int) * T, cudaMemcpyHostToDevice, s));
            if (st == AK_OK) st = set_ctl(m, s0, 0, 0, s0 + 1);
            if (st == AK_OK) st = emit_target(m, b, T, 0, nullptr);
            const int tm = std::min(s0 + T, n - 1) - s0;  // pairs whose next token is still a prompt token
            if (fill_mtp && tm > 0 && st == AK_OK) {
                std::memcpy(m->h_pf + m->pf_cols, prompt + s0 + 1, sizeof(int) * tm);
                st = ck(cudaMemcpyAsync(b.mtok, m->h_pf + m->pf_cols, sizeof(int) * tm, cudaMemcpyHostToDevice, s));
                if (st == AK_OK) st = emit_mtp(m, b, tm, b.hbuf, b.mtok, nullptr);
            }
            *last_col = 0;
        } else {
            const Bufs& b = m->dec;
            std::memcpy(pin_tok(m), prompt + s0, sizeof(int) * T);
            st = ck(cudaMemcpyAsync(b.tok, pin_tok(m), sizeof(int) * T, cudaMemcpyHostToDevice, s));
            if (st == AK_OK) st = set_ctl(m, s0, 0, 0, s0 + 1);
            // a captured graph would bake in the snapshot request, so passes that take one run eagerly
            if (st == AK_OK) st = target(m, T, 0, sr.n ? 0 : use_graph);
            const int tm = std::min(s0 + T, n - 1) - s0;
            if (fill_mtp && tm > 0 && st == AK_OK) {
                std::memcpy(pin_mtok(m), prompt + s0 + 1, sizeof(int) * tm);
                st = ck(cudaMemcpyAsync(b.mtok, pin_mtok(m), sizeof(int) * tm, cudaMemcpyHostToDevice, s));
                if (st == AK_OK) st = mtp(m, tm, b.hbuf, b.mtok, nullptr);
            }
            *last_col = T - 1;
        }
        m->snapreq = SnapReq{};
        if (st == AK_OK) st = ck(cudaStreamSynchronize(s));  // before the pinned staging is reused
        s0 += T;
    }
    m->snapreq = SnapReq{};
    return st;
}

ak_status check_args(ak_q35_model* m, const int32_t* prompt, int n_prompt, int extra) {
    if (!m || !prompt || n_prompt < 1) {
        ak_set_error("qwen35: bad arguments");
        return AK_ERR_INVALID;
    }
    if (n_prompt + extra > m->max_seq) {
        ak_set_error("qwen35: prompt plus generation exceeds max_seq");
        return AK_ERR_INVALID;
    }
    for (int i = 0; i < n_prompt; ++i) {
        if (prompt[i] < 0 || prompt[i] >= m->vocab) {
            ak_set_error("qwen35: prompt token out of range");
            return AK_ERR_INVALID;
        }
    }
    return AK_OK;
}

void fill_stats(ak_q35_model* m, ak_gen_stats* stats, ak_status st, int backend, int n_prompt, int n_gen, float pm,
                float dm) {
    if (!stats) return;
    stats->status = st;
    stats->backend = backend;
    stats->prompt_tokens = n_prompt;
    stats->generated_tokens = n_gen;
    stats->prefill_ms = pm;
    stats->ttft_ms = pm;
    stats->decode_ms = dm;
    stats->prefill_tps = pm > 0 ? n_prompt * 1000.f / pm : 0.f;
    stats->decode_tps = dm > 0 && n_gen > 1 ? (n_gen - 1) * 1000.f / dm : 0.f;
    stats->graph_capture_ms = m->capture_ms;
    stats->n_gpu_layers = m->n_layers;
    stats->vram_peak_mb = m->info.vram_mb;
    stats->weight_vram_mb = static_cast<float>(m->weight_bytes / (1024.0 * 1024.0));
    stats->host_weight_mb = m->info.host_mb;
}

// ------------------------------------------------------------------------------------------------
// Loading
// ------------------------------------------------------------------------------------------------

struct Loader {
    ak_q35_model* m;
    const File& f;
    std::map<std::string, size_t> dev_off;  // tensor name -> offset in m->blob
    size_t* counter = nullptr;              // where mat() adds its bytes

    const Tensor* find(const std::string& name) const {
        auto it = f.tensors.find(name);
        return it == f.tensors.end() ? nullptr : &it->second;
    }
    ak_status mat(const std::string& name, int rows, int K, QMat* out) const {
        const Tensor* t = find(name);
        if (!t) {
            ak_set_error(("qwen35: missing tensor " + name).c_str());
            return AK_ERR_INVALID;
        }
        if (t->dims.size() != 2 || static_cast<int>(t->dims[0]) != K || static_cast<int>(t->dims[1]) != rows) {
            ak_set_error(("qwen35: unexpected shape for " + name).c_str());
            return AK_ERR_INVALID;
        }
        out->type = qcode(t->type);
        if (out->type < 0) {
            ak_set_error(("qwen35: " + name + " must be Q8_0, Q4_K or Q6_K").c_str());
            return AK_ERR_UNSUPPORTED;
        }
        out->w = m->blob + dev_off.at(name);
        out->rows = rows;
        out->K = K;
        if (counter) *counter += out->bytes();
        return AK_OK;
    }
    ak_status vec(const std::string& name, size_t n, const float** out) const {
        const Tensor* t = find(name);
        size_t count = 1;
        if (t) {
            for (uint64_t dd : t->dims) count *= dd;
        }
        if (!t || t->type != kF32 || count != n) {
            ak_set_error(("qwen35: expected f32 tensor " + name).c_str());
            return AK_ERR_INVALID;
        }
        *out = reinterpret_cast<const float*>(m->blob + dev_off.at(name));
        return AK_OK;
    }
};

// Copy the selected tensors from the file into one device allocation, each at a 256-byte boundary.
ak_status upload(ak_q35_model* m, const char* path, const File& f, const std::vector<std::string>& names,
                 std::map<std::string, size_t>* off) {
    std::vector<std::pair<uint64_t, std::string>> order;
    for (const auto& n : names) order.emplace_back(f.tensors.at(n).offset, n);
    std::sort(order.begin(), order.end());
    size_t total = 0;
    for (const auto& o : order) {
        const size_t b = tensor_bytes(f.tensors.at(o.second));
        if (!b) {
            ak_set_error(("qwen35: unsupported tensor type for " + o.second).c_str());
            return AK_ERR_UNSUPPORTED;
        }
        (*off)[o.second] = total;
        total += (b + 255) / 256 * 256;
    }
    void* blob = nullptr;
    ak_status st = dalloc(m, &blob, total);
    if (st != AK_OK) return st;
    m->blob = static_cast<uint8_t*>(blob);
    std::ifstream in(path, std::ios::binary);
    const size_t chunk = 64ull << 20;
    void* pinned = nullptr;
    st = ck(cudaHostAlloc(&pinned, chunk, cudaHostAllocDefault));
    for (size_t i = 0; i < order.size() && st == AK_OK; ++i) {
        const Tensor& t = f.tensors.at(order[i].second);
        const size_t b = tensor_bytes(t);
        in.seekg(static_cast<std::streamoff>(f.data_offset + t.offset));
        for (size_t done = 0; done < b && st == AK_OK;) {
            const size_t n = std::min(chunk, b - done);
            in.read(static_cast<char*>(pinned), static_cast<std::streamsize>(n));
            if (!in) {
                ak_set_error("qwen35: short read of tensor data");
                st = AK_ERR_INVALID;
                break;
            }
            st = ck(cudaMemcpy(m->blob + (*off)[order[i].second] + done, pinned, n, cudaMemcpyHostToDevice));
            done += n;
        }
    }
    if (pinned) cudaFreeHost(pinned);
    return st;
}

ak_status load_attn(ak_q35_model* m, const Loader& ld, const std::string& p, Layer& L) {
    const size_t kv = static_cast<size_t>(m->max_seq) * m->hkv * kHd;
    ak_status st = ld.mat(p + "attn_q.weight", m->hq * kHd * 2, m->d, &L.wq);
    if (st == AK_OK) st = ld.mat(p + "attn_k.weight", m->hkv * kHd, m->d, &L.wk);
    if (st == AK_OK) st = ld.mat(p + "attn_v.weight", m->hkv * kHd, m->d, &L.wv);
    if (st == AK_OK) st = ld.mat(p + "attn_output.weight", m->d, m->hq * kHd, &L.wo);
    if (st == AK_OK) st = ld.vec(p + "attn_q_norm.weight", kHd, &L.q_norm);
    if (st == AK_OK) st = ld.vec(p + "attn_k_norm.weight", kHd, &L.k_norm);
    if (st == AK_OK) st = dalloc_n(m, &L.kc, kv);
    if (st == AK_OK) st = dalloc_n(m, &L.vc, kv);
    return st;
}

ak_status load_ffn(ak_q35_model* m, const Loader& ld, const std::string& p, Layer& L) {
    ak_status st = ld.vec(p + "attn_norm.weight", m->d, &L.attn_norm);
    if (st == AK_OK) st = ld.vec(p + "post_attention_norm.weight", m->d, &L.post_norm);
    if (st == AK_OK) st = ld.mat(p + "ffn_gate.weight", m->ff, m->d, &L.wgate);
    if (st == AK_OK) st = ld.mat(p + "ffn_up.weight", m->ff, m->d, &L.wup);
    if (st == AK_OK) st = ld.mat(p + "ffn_down.weight", m->d, m->ff, &L.wdown);
    return st;
}

}  // namespace

extern "C" ak_status ak_q35_destroy(ak_q35_model* m) {
    if (!m) return AK_OK;
    if (m->stream) cudaStreamSynchronize(m->stream);
    for (auto& kv : m->graphs) cudaGraphExecDestroy(kv.second);
    for (void* p : m->mem) cudaFree(p);
    if (m->embd_host) cudaFreeHost(m->embd_host);
    if (m->h_pin) cudaFreeHost(m->h_pin);
    if (m->h_pf) cudaFreeHost(m->h_pf);
    if (m->blas) cublasDestroy(m->blas);
    if (m->stream) cudaStreamDestroy(m->stream);
    delete m;
    return AK_OK;
}

extern "C" ak_status ak_q35_load(const char* path, int max_seq, ak_q35_model** out, ak_q35_info* info) {
    if (!path || !out) {
        ak_set_error("qwen35 load: bad arguments");
        return AK_ERR_INVALID;
    }
    *out = nullptr;
    ak_status st = ak_init(0);
    if (st != AK_OK) return st;
    File f;
    st = parse(path, &f);
    if (st != AK_OK) return st;
    const std::string arch = f.str.count("general.architecture") ? f.str["general.architecture"] : "";
    if (arch != "qwen35") {
        ak_set_error(("qwen35: architecture '" + arch + "' is not qwen35").c_str());
        return AK_ERR_UNSUPPORTED;
    }
    auto num = [&](const char* key, double fb) {
        auto it = f.num.find(arch + "." + key);
        return it == f.num.end() ? fb : it->second;
    };
    ak_q35_model* m = new ak_q35_model();
    const int blocks = static_cast<int>(num("block_count", 0));
    const int nextn = static_cast<int>(num("nextn_predict_layers", 0));
    m->n_layers = blocks - nextn;
    m->d = static_cast<int>(num("embedding_length", 0));
    m->ff = static_cast<int>(num("feed_forward_length", 0));
    m->hq = static_cast<int>(num("attention.head_count", 0));
    m->hkv = static_cast<int>(num("attention.head_count_kv", 0));
    const int key_len = static_cast<int>(num("attention.key_length", 0));
    const int val_len = static_cast<int>(num("attention.value_length", 0));
    m->n_rot = static_cast<int>(num("rope.dimension_count", key_len));
    m->eps = static_cast<float>(num("attention.layer_norm_rms_epsilon", 1e-6));
    m->theta = static_cast<float>(num("rope.freq_base", 1e7));
    m->hk = static_cast<int>(num("ssm.group_count", 0));
    m->hv = static_cast<int>(num("ssm.time_step_rank", 0));
    const int d_state = static_cast<int>(num("ssm.state_size", 0));
    m->d_inner = static_cast<int>(num("ssm.inner_size", 0));
    const int conv_k = static_cast<int>(num("ssm.conv_kernel", 0));
    const int interval = static_cast<int>(num("full_attention_interval", 4));
    m->conv_ch = 2 * m->hk * d_state + m->d_inner;
    m->max_seq = max_seq > 0 ? max_seq : 4096;
    m->n_split = (m->max_seq + kAttnChunk - 1) / kAttnChunk;
    const Tensor* te = f.tensors.count("token_embd.weight") ? &f.tensors["token_embd.weight"] : nullptr;
    m->vocab = te && te->dims.size() == 2 ? static_cast<int>(te->dims[1]) : 0;
    const int ratio = m->hkv ? m->hq / m->hkv : 0;
    if (m->n_layers <= 0 || m->d % 256 || m->ff % 256 || key_len != kHd || val_len != kHd || d_state != kDk ||
        m->d_inner != m->hv * kDk || conv_k != 4 || m->n_rot != 64 || m->hk <= 0 || m->hv % m->hk ||
        (ratio != 1 && ratio != 2 && ratio != 4) || m->hq % m->hkv || !te || interval <= 0) {
        delete m;
        ak_set_error("qwen35: unsupported shape (needs head 256, rope 64, ssm state 128, conv 4)");
        return AK_ERR_UNSUPPORTED;
    }
    m->mtp.present = nextn == 1 && f.tensors.count("blk." + std::to_string(m->n_layers) + ".nextn.eh_proj.weight");
    st = ck(cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking));

    // Everything except the embedding table goes to the device (the MTP block only if it is complete).
    std::vector<std::string> names;
    for (const auto& kv : f.tensors) {
        const std::string& n = kv.first;
        if (n == "token_embd.weight") continue;
        if (n.rfind("blk.", 0) == 0) {
            const int b = std::stoi(n.substr(4));
            if (b > m->n_layers || (b == m->n_layers && !m->mtp.present)) continue;
            if (b == m->n_layers && n.find("nextn.embed_tokens") != std::string::npos) continue;
            if (b == m->n_layers && n.find("nextn.shared_head_head") != std::string::npos) continue;
        }
        names.push_back(n);
    }
    Loader ld{m, f, {}, &m->weight_bytes};
    if (st == AK_OK) st = upload(m, path, f, names, &ld.dev_off);

    // Embedding table: pinned host memory mapped into the device address space.
    if (st == AK_OK) {
        m->embd_type = qcode(te->type);
        const size_t eb = tensor_bytes(*te);
        if (m->embd_type < 0 || !eb) {
            ak_set_error("qwen35: token_embd must be Q8_0, Q4_K or Q6_K");
            st = AK_ERR_UNSUPPORTED;
        }
        if (st == AK_OK) st = ck(cudaHostAlloc(reinterpret_cast<void**>(&m->embd_host), eb, cudaHostAllocMapped));
        if (st == AK_OK) {
            std::ifstream in(path, std::ios::binary);
            in.seekg(static_cast<std::streamoff>(f.data_offset + te->offset));
            in.read(reinterpret_cast<char*>(m->embd_host), static_cast<std::streamsize>(eb));
            if (!in) {
                ak_set_error("qwen35: short read of token_embd");
                st = AK_ERR_INVALID;
            }
        }
        void* dp = nullptr;
        if (st == AK_OK) st = ck(cudaHostGetDevicePointer(&dp, m->embd_host, 0));
        m->embd_dev = static_cast<const uint8_t*>(dp);
        m->info.host_mb = static_cast<float>(eb / (1024.0 * 1024.0));
    }

    if (st == AK_OK) st = ld.vec("output_norm.weight", m->d, &m->out_norm);
    if (st == AK_OK) {
        if (f.tensors.count("output.weight")) {
            st = ld.mat("output.weight", m->vocab, m->d, &m->out);
        } else {
            ak_set_error("qwen35: tied output embedding is not supported (embedding table is host-resident)");
            st = AK_ERR_UNSUPPORTED;
        }
    }
    m->layers.resize(m->n_layers);
    int n_attn = 0;
    for (int l = 0; l < m->n_layers && st == AK_OK; ++l) {
        Layer& L = m->layers[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        L.linear = ((l + 1) % interval) != 0;
        st = load_ffn(m, ld, p, L);
        if (L.linear) {
            if (st == AK_OK) st = ld.mat(p + "attn_qkv.weight", m->conv_ch, m->d, &L.wqkv);
            if (st == AK_OK) st = ld.mat(p + "attn_gate.weight", m->d_inner, m->d, &L.wz);
            if (st == AK_OK) st = ld.mat(p + "ssm_alpha.weight", m->hv, m->d, &L.walpha);
            if (st == AK_OK) st = ld.mat(p + "ssm_beta.weight", m->hv, m->d, &L.wbeta);
            if (st == AK_OK) st = ld.mat(p + "ssm_out.weight", m->d, m->d_inner, &L.wout);
            if (st == AK_OK) st = ld.vec(p + "ssm_conv1d.weight", static_cast<size_t>(m->conv_ch) * 4, &L.conv_w);
            if (st == AK_OK) st = ld.vec(p + "ssm_dt.bias", m->hv, &L.dt_bias);
            if (st == AK_OK) st = ld.vec(p + "ssm_a", m->hv, &L.ssm_a);
            if (st == AK_OK) st = ld.vec(p + "ssm_norm.weight", kDk, &L.ssm_norm);
            const size_t col_n = static_cast<size_t>(m->hv) * kDk;
            if (st == AK_OK) st = dalloc_n(m, &L.conv_state, static_cast<size_t>(3) * m->conv_ch);
            if (st == AK_OK) st = dalloc_n(m, &L.gdn_state, col_n * kDk);
            if (st == AK_OK) st = dalloc_n(m, &L.conv_save, static_cast<size_t>(2) * kMaxCols * m->conv_ch);
            if (st == AK_OK) st = dalloc_n(m, &L.k_save, 2 * kMaxCols * col_n);
            if (st == AK_OK) st = dalloc_n(m, &L.v_save, 2 * kMaxCols * col_n);
            if (st == AK_OK) st = dalloc_n(m, &L.db_save, static_cast<size_t>(2) * kMaxCols * 2 * m->hv);
        } else {
            ++n_attn;
            if (st == AK_OK) st = load_attn(m, ld, p, L);
        }
    }
    if (st == AK_OK && m->mtp.present) {
        Mtp& M = m->mtp;
        const std::string p = "blk." + std::to_string(m->n_layers) + ".";
        ld.counter = &m->mtp_bytes;
        M.blk.linear = false;
        st = load_ffn(m, ld, p, M.blk);
        if (st == AK_OK) st = load_attn(m, ld, p, M.blk);
        if (st == AK_OK) st = ld.mat(p + "nextn.eh_proj.weight", m->d, 2 * m->d, &M.eh_proj);
        if (st == AK_OK) st = ld.vec(p + "nextn.enorm.weight", m->d, &M.enorm);
        if (st == AK_OK) st = ld.vec(p + "nextn.hnorm.weight", m->d, &M.hnorm);
        if (st == AK_OK) {
            const std::string hn = p + "nextn.shared_head_norm.weight";
            st = ld.vec(f.tensors.count(hn) ? hn : "output_norm.weight", m->d, &M.head_norm);
        }
    }
    const size_t C = kMaxCols;
    m->big_n = std::max(m->conv_ch + m->d_inner, m->hq * kHd * 2 + m->hkv * kHd);
    m->aq_n = std::max(m->d_inner, m->hq * kHd);
    if (m->aq_n != m->d_inner || m->aq_n != m->hq * kHd) {
        // Q8_1 activations for wout / wo use the matrix K as column stride; both are aq_n here.
        ak_set_error("qwen35: ssm inner size must equal heads * head_dim");
        st = st == AK_OK ? AK_ERR_UNSUPPORTED : st;
    }
    if (st == AK_OK) st = alloc_bufs(m, &m->dec, kMaxCols, false);
    if (st == AK_OK) st = dalloc_n(m, &m->logits, C * m->vocab);
    if (st == AK_OK) st = dalloc_n(m, &m->part, C * m->hq * m->n_split * (kHd + 2));
    if (st == AK_OK) st = dalloc_n(m, &m->best, C + 1);
    if (st == AK_OK) st = dalloc_n(m, &m->d_arg, C);
    if (st == AK_OK) st = dalloc_n(m, &m->d_ctl, kCtlN);
    if (st == AK_OK) st = dalloc_n(m, &m->d_count, 1);
    if (st == AK_OK) st = dalloc_n(m, &m->d_out, static_cast<size_t>(m->max_seq) + 1);
    if (st == AK_OK) st = ck(cudaHostAlloc(reinterpret_cast<void**>(&m->h_pin), sizeof(int) * (kCtlN + 4 * C), 0));
    {
        const char* e = std::getenv("AK_Q35_NO_PF");
        m->no_pf = e && e[0] == '1';
        m->draft_vocab = m->vocab;
        const char* a = std::getenv("AK_Q35_ATTN_OLD");
        if (a && a[0] == '1') m->attn = 0;
        const char* av = std::getenv("AK_Q35_ATTN");
        if (av && av[0] >= '0' && av[0] <= '2') m->attn = av[0] - '0';
    }
    // The tensor-core attention kernels need more than the default 48 KB of dynamic shared memory.
    for (const void* f : {reinterpret_cast<const void*>(k_attn_mma<1, false>),
                          reinterpret_cast<const void*>(k_attn_mma<2, false>),
                          reinterpret_cast<const void*>(k_attn_mma<4, false>),
                          reinterpret_cast<const void*>(k_attn_mma<1, true>),
                          reinterpret_cast<const void*>(k_attn_mma<2, true>),
                          reinterpret_cast<const void*>(k_attn_mma<4, true>),
                          reinterpret_cast<const void*>(k_attn_mma2<1, false>),
                          reinterpret_cast<const void*>(k_attn_mma2<2, false>),
                          reinterpret_cast<const void*>(k_attn_mma2<4, false>),
                          reinterpret_cast<const void*>(k_attn_mma2<1, true>),
                          reinterpret_cast<const void*>(k_attn_mma2<2, true>),
                          reinterpret_cast<const void*>(k_attn_mma2<4, true>)}) {
        if (st == AK_OK) {
            st = ck(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kAttnSmem)));
        }
    }
    if (st == AK_OK) st = reset(m);
    if (st != AK_OK) {
        ak_q35_destroy(m);
        return st;
    }
    ak_q35_info& i = m->info;
    std::snprintf(i.arch, sizeof(i.arch), "%s", arch.c_str());
    i.n_layers = m->n_layers;
    i.n_attn_layers = n_attn;
    i.d_model = m->d;
    i.d_ff = m->ff;
    i.vocab = m->vocab;
    i.n_heads = m->hq;
    i.n_kv_heads = m->hkv;
    i.head_dim = kHd;
    i.rope_dims = m->n_rot;
    i.ssm_k_heads = m->hk;
    i.ssm_v_heads = m->hv;
    i.ssm_state = kDk;
    i.max_seq = m->max_seq;
    i.rope_theta = m->theta;
    i.rms_eps = m->eps;
    i.weight_gb_per_token = static_cast<float>(m->weight_bytes / 1.0e9);
    i.vram_mb = static_cast<float>(m->dev_bytes / (1024.0 * 1024.0));
    i.has_mtp = m->mtp.present ? 1 : 0;
    i.mtp_gb_per_draft = static_cast<float>((m->mtp_bytes + m->out.bytes()) / 1.0e9);
    i.max_cols = kMaxCols;
    if (info) *info = i;
    *out = m;
    return AK_OK;
}

extern "C" ak_status ak_q35_generate(ak_q35_model* m, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                     int n_gen, int use_graph, ak_gen_stats* stats) {
    ak_status st = check_args(m, prompt, n_prompt, n_gen - 1);
    if (st != AK_OK) return st;
    if (n_gen < 1 || !out_tokens) {
        ak_set_error("qwen35 generate: bad arguments");
        return AK_ERR_INVALID;
    }
    if (stats) std::memset(stats, 0, sizeof(*stats));
    st = reset(m);
    const auto t0 = std::chrono::steady_clock::now();
    int lc = 0;
    if (st == AK_OK) st = prefill(m, prompt, n_prompt, use_graph, false, &lc);
    int first = 0;
    if (st == AK_OK) st = ck(cudaMemcpy(&first, m->d_arg + lc, sizeof(int), cudaMemcpyDeviceToHost));
    const auto t1 = std::chrono::steady_clock::now();
    out_tokens[0] = first;
    if (n_gen > 1 && st == AK_OK) {
        pin_tok(m)[0] = first;
        st = ck(cudaMemcpyAsync(m->dec.tok, pin_tok(m), sizeof(int), cudaMemcpyHostToDevice, m->stream));
        if (st == AK_OK) st = set_ctl(m, n_prompt, 0, 0, 0);
        for (int g = 1; g < n_gen && st == AK_OK; ++g) st = target_plain(m, use_graph);
        if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
        if (st == AK_OK) st = ck(cudaMemcpy(out_tokens + 1, m->d_out, sizeof(int) * (n_gen - 1), cudaMemcpyDeviceToHost));
    }
    const auto t2 = std::chrono::steady_clock::now();
    fill_stats(m, stats, st, use_graph ? AK_BACKEND_CUDA_GRAPH : AK_BACKEND_EAGER, n_prompt, n_gen,
               std::chrono::duration<float, std::milli>(t1 - t0).count(),
               std::chrono::duration<float, std::milli>(t2 - t1).count());
    return st;
}

namespace {

constexpr float kLaPrior = 0.9f;
constexpr float kLaPull = 0.02f;

// Speculation cost model and acceptance estimates, updated every round from GPU event timings.
struct Policy {
    float acc[kMaxCols];                // P(MTP draft i accepted | draft i - 1 accepted), i >= 1
    // Prompt-lookup acceptance per position. Only lookup rounds update it, so a pessimistic start meant lookup was
    // never tried once MTP acceptance was high, and whether it ever got going (1.6x on code-copying output) came
    // down to timing noise in the first rounds. Start optimistic (an n-gram match in code usually continues), and
    // pull it back toward that prior in rounds where a match exists but MTP was chosen (kLaPrior / kLaPull).
    float la = kLaPrior;
    int ng_len = 4;                     // current lookup proposal cap
    float verify_ms[kMaxCols + 1] = {};
    float first_ms = -1.f;              // MTP pass over the accepted pairs + head (first draft)
    float step_ms = -1.f;               // each chained draft
    float fill_ms = -1.f;               // MTP pass over the accepted pairs without the head (lookup rounds)
};

void ema(float* v, float x) { *v = *v <= 0.f ? x : 0.8f * *v + 0.2f * x; }

float est_verify(const Policy& p, int T) {
    if (p.verify_ms[T] > 0.f) return p.verify_ms[T];
    for (int dd = 1; dd < kMaxCols; ++dd) {
        for (int t : {T - dd, T + dd}) {
            if (t >= 1 && t <= kMaxCols && p.verify_ms[t] > 0.f) return p.verify_ms[t] * (1.f + 0.03f * (T - t));
        }
    }
    return 25.f;
}

// Expected tokens from one round: the target's own token plus the accepted prefix of k drafts.
float expect_mtp(const Policy& p, int k) {
    float e = 1.f, run = 1.f;
    for (int i = 1; i <= k; ++i) {
        run *= p.acc[i];
        e += run;
    }
    return e;
}

float expect_geo(float a, int k) {
    float e = 1.f, run = 1.f;
    for (int i = 1; i <= k; ++i) {
        run *= a;
        e += run;
    }
    return e;
}

// Longest n in [n_min, n_max] such that the last n tokens of h occur earlier; returns the index of the token that
// followed the most recent such occurrence, or -1.
int lookup(const std::vector<int>& h, int n_min, int n_max) {
    const int L = static_cast<int>(h.size());
    for (int n = std::min(n_max, L - 1); n >= n_min; --n) {
        const int* suf = h.data() + L - n;
        for (int s = L - n - 1; s >= 0; --s) {
            if (std::equal(suf, suf + n, h.data() + s)) return s + n;
        }
    }
    return -1;
}

ak_status spec_run(ak_q35_model* m, const int32_t* prompt, int n_prompt, int32_t* out_tokens, int n_gen,
                   const ak_q35_spec_opts& o, ak_gen_stats* stats, ak_q35_spec_stats* spec,
                   const ak_q35_run_opts* run = nullptr, int32_t* n_out = nullptr) {
    if (n_out) *n_out = 0;
    ak_status st = check_args(m, prompt, n_prompt, n_gen + kMaxCols);
    if (st != AK_OK) return st;
    if (n_gen < 1 || !out_tokens || !m->mtp.present) {
        ak_set_error(!m->mtp.present ? "qwen35 spec: model has no MTP block" : "qwen35 spec: bad arguments");
        return AK_ERR_INVALID;
    }
    const int start = run ? run->start : 0;
    if (start < 0 || start >= n_prompt) {
        ak_set_error("qwen35 spec: run.start must be in [0, n_prompt)");
        return AK_ERR_INVALID;
    }
    // Snapshot points inside the prompt, in order: {position, slot}.
    std::vector<std::pair<int, int>> snaps;
    if (run) {
        for (int i = 0; i < AK_Q35_CACHE_SLOTS && run->snap_at[i] > 0; ++i) {
            const int at = run->snap_at[i];
            const int slot = run->snap_slot[i];
            if (at <= start || at >= n_prompt || (!snaps.empty() && at <= snaps.back().first)) continue;
            if (slot < 0 || slot >= AK_Q35_CACHE_SLOTS) {
                ak_set_error("qwen35 spec: snapshot slot out of range");
                return AK_ERR_INVALID;
            }
            snaps.emplace_back(at, slot);
        }
    }
    ak_q35_token_cb cb = run ? run->cb : nullptr;
    void* cb_user = run ? run->user : nullptr;
    const int max_k = std::max(1, std::min(o.max_draft > 0 ? o.max_draft : kMaxCols - 1, kMaxCols - 1));
    const int ng_min = std::max(1, o.ngram_min > 0 ? o.ngram_min : 2);
    const int ng_max = std::max(ng_min, o.ngram_max > 0 ? o.ngram_max : 6);
    const bool use_lookup = !o.fixed && !o.no_lookup;
    m->draft_vocab = o.draft_vocab > 0 ? std::min(o.draft_vocab, m->vocab) : std::min(65536, m->vocab);
    if (stats) std::memset(stats, 0, sizeof(*stats));
    if (spec) std::memset(spec, 0, sizeof(*spec));
    if (start == 0) {
        st = reset(m);
    } else {  // state restored by the caller; only clear the argmax scratch
        st = ck(cudaMemsetAsync(m->best, 0, sizeof(unsigned long long) * (kMaxCols + 1), m->stream));
    }
    const auto t0 = std::chrono::steady_clock::now();
    int lc = 0;
    // One prefill over [start, n_prompt); the snapshots are taken inside whichever pass covers their position.
    // (Before, each snapshot ended a segment, so every turn paid one extra pass - all weights read once, ~22 ms -
    // for the few tokens after the last snapshot, plus a GEMV pass per short segment tail.)
    if (st == AK_OK) st = prefill(m, prompt, n_prompt, 1, true, &lc, start, n_prompt, &snaps);
    int* res = pin_res(m);
    if (st == AK_OK) st = ck(cudaMemcpy(res, m->d_arg + lc, sizeof(int), cudaMemcpyDeviceToHost));
    const auto t1 = std::chrono::steady_clock::now();
    Bufs& D = m->dec;
    int g = res[0];
    int emitted = 0;
    out_tokens[emitted++] = g;
    int reported = 0;
    bool stop = false;
    auto report = [&] {
        if (cb && emitted > reported) {
            stop = cb(cb_user, out_tokens + reported, emitted - reported) != 0;
            reported = emitted;
        }
    };
    if (st == AK_OK) report();
    std::vector<int> h(prompt, prompt + n_prompt);  // h[i]: token at position i; the last one is g, not yet processed
    h.push_back(g);
    int p = n_prompt;  // positions the target has processed
    int commit = 0;    // columns of the last verify pass to replay into the recurrent state
    int par = 0;       // save buffer the last pass wrote
    int n_acc = 1;     // columns of hbuf / d_arg that hold the newly accepted pairs
    int hcol = lc;     // first such column
    Policy pol;
    for (float& a : pol.acc) a = 0.8f;
    cudaEvent_t ev[4];
    for (auto& e : ev) cudaEventCreate(&e);
    int rounds = 0, drafted = 0, accepted = 0, l_rounds = 0, l_drafted = 0, l_accepted = 0;
    while (emitted < n_gen && st == AK_OK && !stop) {
        const int room = n_gen - emitted - 1;  // drafts that can still be emitted
        int k = max_k;
        bool from_lookup = false;
        int at = -1;
        if (!o.fixed) {
            const float first = pol.first_ms > 0.f ? pol.first_ms : 3.f;
            const float step = pol.step_ms > 0.f ? pol.step_ms : 2.f;
            float best = 0.f;
            for (int kk = 1; kk <= max_k; ++kk) {
                const float r = expect_mtp(pol, kk) / (first + (kk - 1) * step + est_verify(pol, kk + 1));
                if (r > best) {
                    best = r;
                    k = kk;
                }
            }
            if (use_lookup) {
                at = lookup(h, ng_min, ng_max);
                if (at >= 0) {
                    const int L = std::min({pol.ng_len, max_k, static_cast<int>(h.size()) - at});
                    const float fill = pol.fill_ms > 0.f ? pol.fill_ms : 1.f;
                    if (expect_geo(pol.la, L) / (fill + est_verify(pol, L + 1)) > best) {
                        from_lookup = true;
                        k = L;
                    } else {
                        pol.la += kLaPull * (kLaPrior - pol.la);
                    }
                }
            }
        }
        k = std::max(0, std::min(k, room));
        int* pt = pin_tok(m);
        pt[0] = g;
        if (from_lookup) {
            for (int i = 0; i < k; ++i) pt[1 + i] = h[at + i];
        }
        st = ck(cudaMemcpyAsync(D.tok, pt, sizeof(int) * (from_lookup ? k + 1 : 1), cudaMemcpyHostToDevice, m->stream));
        if (st == AK_OK) st = set_ctl(m, p, commit, par, p - n_acc + 1);
        const float* hsrc = D.hbuf + static_cast<size_t>(hcol) * m->d;
        const bool mtp_draft = !from_lookup && k > 0;
        cudaEventRecord(ev[0], m->stream);
        // MTP over the pairs (h, accepted token) at positions p - n_acc + 1 .. p: fills its KV cache and, for MTP
        // rounds, drafts d1 from the last pair; further drafts chain on the MTP output.
        if (st == AK_OK) st = mtp(m, n_acc, hsrc, m->d_arg + hcol, mtp_draft ? D.tok + 1 : nullptr);
        cudaEventRecord(ev[1], m->stream);
        for (int j = 2; mtp_draft && j <= k && st == AK_OK; ++j) {
            const int prevT = j == 2 ? n_acc : 1;
            st = mtp(m, 1, D.mh + static_cast<size_t>(prevT - 1) * m->d, D.tok + j - 1, D.tok + j);
        }
        cudaEventRecord(ev[2], m->stream);
        // Verify [g, d1..dk] at positions p .. p + k.
        if (st == AK_OK) st = target(m, k + 1, 1, 1);
        cudaEventRecord(ev[3], m->stream);
        if (st == AK_OK) st = ck(cudaMemcpyAsync(res, m->d_arg, sizeof(int) * (k + 1), cudaMemcpyDeviceToHost, m->stream));
        if (st == AK_OK && mtp_draft) {
            st = ck(cudaMemcpyAsync(res + kMaxCols, D.tok, sizeof(int) * (k + 1), cudaMemcpyDeviceToHost, m->stream));
        }
        if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
        if (st != AK_OK) break;
        if (from_lookup) {
            for (int i = 0; i <= k; ++i) res[kMaxCols + i] = pt[i];
        }
        float t_first = 0.f, t_chain = 0.f, t_ver = 0.f;
        cudaEventElapsedTime(&t_first, ev[0], ev[1]);
        cudaEventElapsedTime(&t_chain, ev[1], ev[2]);
        cudaEventElapsedTime(&t_ver, ev[2], ev[3]);
        ema(&pol.verify_ms[k + 1], t_ver);
        if (mtp_draft) {
            ema(&pol.first_ms, t_first);
            if (k > 1) ema(&pol.step_ms, t_chain / (k - 1));
        } else {
            ema(&pol.fill_ms, t_first);
        }
        int n = 1;
        while (n <= k && res[kMaxCols + n] == res[n - 1]) ++n;
        for (int i = 0; i < n; ++i) {
            if (emitted < n_gen) out_tokens[emitted++] = res[i];
            h.push_back(res[i]);
        }
        const int got = n - 1;  // accepted drafts
        if (from_lookup) {
            ++l_rounds;
            l_drafted += k;
            l_accepted += got;
            pol.ng_len = got == k ? std::min(kMaxCols - 1, pol.ng_len + 2) : std::max(2, got + 1);
            for (int i = 0; i < std::min(got + 1, k); ++i) pol.la = 0.9f * pol.la + (i < got ? 0.1f : 0.f);
        } else if (k > 0) {
            for (int i = 1; i <= std::min(got + 1, k); ++i) pol.acc[i] = 0.9f * pol.acc[i] + (i <= got ? 0.1f : 0.f);
        }
        ++rounds;
        drafted += k;
        accepted += got;
        g = res[n - 1];
        p += n;
        commit = n;
        par ^= 1;
        n_acc = n;
        hcol = 0;
        report();
    }
    for (auto& e : ev) cudaEventDestroy(e);
    const auto t2 = std::chrono::steady_clock::now();
    if (n_out) *n_out = emitted;
    fill_stats(m, stats, st, AK_BACKEND_CUDA_GRAPH, n_prompt - start, emitted,
               std::chrono::duration<float, std::milli>(t1 - t0).count(),
               std::chrono::duration<float, std::milli>(t2 - t1).count());
    if (spec) {
        spec->rounds = rounds;
        spec->drafted = drafted;
        spec->accepted = accepted;
        spec->acceptance = drafted ? static_cast<float>(accepted) / drafted : 0.f;
        spec->tokens_per_round = rounds ? static_cast<float>(emitted - 1) / rounds : 0.f;
        spec->lookup_rounds = l_rounds;
        spec->lookup_drafted = l_drafted;
        spec->lookup_accepted = l_accepted;
        spec->mean_draft = rounds ? static_cast<float>(drafted) / rounds : 0.f;
    }
    return st;
}

}  // namespace

// Greedy speculative decoding with the MTP head drafting n_draft tokens per round over the full vocabulary (unless
// AK_Q35_DRAFT_VOCAB is set); output equals plain greedy.
extern "C" ak_status ak_q35_generate_mtp(ak_q35_model* m, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                         int n_gen, int n_draft, ak_gen_stats* stats, ak_q35_spec_stats* spec) {
    if (!m || n_draft < 1 || n_draft > kMaxCols - 1) {
        ak_set_error("qwen35 mtp: bad arguments (n_draft 1..11)");
        return AK_ERR_INVALID;
    }
    ak_q35_spec_opts o{};
    o.max_draft = n_draft;
    o.fixed = 1;
    const char* dv = std::getenv("AK_Q35_DRAFT_VOCAB");
    o.draft_vocab = dv ? std::atoi(dv) : m->vocab;
    return spec_run(m, prompt, n_prompt, out_tokens, n_gen, o, stats, spec);
}

extern "C" ak_status ak_q35_generate_spec(ak_q35_model* m, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                          int n_gen, const ak_q35_spec_opts* opts, ak_gen_stats* stats,
                                          ak_q35_spec_stats* spec) {
    if (!m) {
        ak_set_error("qwen35 spec: null model");
        return AK_ERR_INVALID;
    }
    ak_q35_spec_opts o{};
    if (opts) o = *opts;
    return spec_run(m, prompt, n_prompt, out_tokens, n_gen, o, stats, spec);
}

extern "C" ak_status ak_q35_generate_spec_ex(ak_q35_model* m, const int32_t* prompt, int n_prompt,
                                             int32_t* out_tokens, int n_gen, const ak_q35_spec_opts* opts,
                                             const ak_q35_run_opts* run, ak_gen_stats* stats,
                                             ak_q35_spec_stats* spec, int32_t* n_out) {
    if (!m) {
        ak_set_error("qwen35 spec: null model");
        return AK_ERR_INVALID;
    }
    ak_q35_spec_opts o{};
    if (opts) o = *opts;
    return spec_run(m, prompt, n_prompt, out_tokens, n_gen, o, stats, spec, run, n_out);
}

extern "C" ak_status ak_q35_cache_load(ak_q35_model* m, int slot) {
    if (!m) {
        ak_set_error("qwen35 cache: null model");
        return AK_ERR_INVALID;
    }
    return snap_copy(m, slot, false);
}

extern "C" ak_status ak_q35_set_splitk(ak_q35_model* m, int max_split) {
    if (!m) {
        ak_set_error("qwen35 set_splitk: null model");
        return AK_ERR_INVALID;
    }
    m->splitk_max = std::max(1, std::min(max_split, 16));
    return AK_OK;
}

extern "C" ak_status ak_q35_cache_save(ak_q35_model* m, int slot) {
    if (!m) {
        ak_set_error("qwen35 cache: null model");
        return AK_ERR_INVALID;
    }
    return snap_copy(m, slot, true);
}

extern "C" ak_status ak_q35_cache_read(ak_q35_model* m, int slot, float* out, size_t n, size_t* n_total) {
    if (!m || slot < 0 || slot >= AK_Q35_CACHE_SLOTS) {
        ak_set_error("qwen35 cache read: bad arguments");
        return AK_ERR_INVALID;
    }
    size_t n_lin = 0;
    for (const Layer& L : m->layers) n_lin += L.linear ? 1 : 0;
    const size_t total = n_lin * (3 * static_cast<size_t>(m->conv_ch) + static_cast<size_t>(m->hv) * kDk * kDk);
    if (n_total) *n_total = total;
    if (!out || n == 0) return AK_OK;
    if (!m->snap[slot]) {
        ak_set_error("qwen35 cache read: slot is empty");
        return AK_ERR_INVALID;
    }
    return ck(cudaMemcpy(out, m->snap[slot], sizeof(float) * std::min(n, total), cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_q35_set_attn(ak_q35_model* m, int tensor_core) {
    if (!m) {
        ak_set_error("qwen35 set_attn: null model");
        return AK_ERR_INVALID;
    }
    const int mode = tensor_core < 0 || tensor_core > 2 ? 2 : tensor_core;
    if (m->attn == mode) return AK_OK;
    ak_status st = ck(cudaStreamSynchronize(m->stream));
    for (auto& kv : m->graphs) cudaGraphExecDestroy(kv.second);  // captured graphs bake in the kernel choice
    m->graphs.clear();
    m->attn = mode;
    return st;
}

extern "C" ak_status ak_q35_set_int8(ak_q35_model* m, int on, int max_cols) {
    if (!m) {
        ak_set_error("qwen35 set_int8: null model");
        return AK_ERR_INVALID;
    }
    m->int8_gemm = on != 0;
    m->int8_max = max_cols < 0 ? (1 << 30) : max_cols;
    return AK_OK;
}

extern "C" ak_status ak_q35_set_qgemm(ak_q35_model* m, int max_cols) {
    if (!m) {
        ak_set_error("qwen35 set_qgemm: null model");
        return AK_ERR_INVALID;
    }
    m->qgemm_max = max_cols < 0 ? (1 << 30) : max_cols;
    return AK_OK;
}

extern "C" ak_status ak_q35_set_prefill(ak_q35_model* m, int batched) {
    if (!m) {
        ak_set_error("qwen35 set_prefill: null model");
        return AK_ERR_INVALID;
    }
    m->no_pf = !batched;
    return AK_OK;
}

extern "C" ak_status ak_q35_logits(ak_q35_model* m, const int32_t* prompt, int n_prompt, int use_graph,
                                   float* logits) {
    ak_status st = check_args(m, prompt, n_prompt, 0);
    if (st != AK_OK) return st;
    if (!logits) {
        ak_set_error("qwen35 logits: null output");
        return AK_ERR_INVALID;
    }
    st = reset(m);
    int lc = 0;
    if (st == AK_OK) st = prefill(m, prompt, n_prompt, use_graph, false, &lc);
    if (st == AK_OK) {
        st = ck(cudaMemcpy(logits, m->logits + static_cast<size_t>(lc) * m->vocab, sizeof(float) * m->vocab,
                           cudaMemcpyDeviceToHost));
    }
    return st;
}

// GPU time of one eager T = 1 decode step by kernel kind (kKCount buckets, see Kind above). Runs at position 0
// with the current state; meant for timing only.
extern "C" ak_status ak_q35_profile(ak_q35_model* m, float* ms_by_kind, int32_t* count_by_kind, int n_kinds) {
    return ak_q35_profile_prefill(m, 1, 0, ms_by_kind, count_by_kind, n_kinds);
}

// GPU time by kernel kind of one pass of T columns at position pos (T > max_cols: one batched prefill chunk on the
// prefill buffers). Token ids and the state are whatever the buffers hold; meant for timing only.
extern "C" ak_status ak_q35_profile_prefill(ak_q35_model* m, int T, int pos, float* ms_by_kind,
                                            int32_t* count_by_kind, int n_kinds) {
    if (!m || !ms_by_kind || !count_by_kind || n_kinds < kKCount || T < 1 || pos < 0 || pos + T > m->max_seq) {
        ak_set_error("qwen35 profile: bad arguments");
        return AK_ERR_INVALID;
    }
    for (int i = 0; i < n_kinds; ++i) {
        ms_by_kind[i] = 0.f;
        count_by_kind[i] = 0;
    }
    ak_status st = AK_OK;
    const bool pf = T > kMaxCols;
    if (pf) {
        st = ensure_pf(m);
        if (st == AK_OK && T > m->pf_cols) {
            ak_set_error("qwen35 profile: T exceeds the prefill chunk length");
            return AK_ERR_INVALID;
        }
        if (st == AK_OK) st = ck(cudaMemsetAsync(m->pf.tok, 0, sizeof(int) * T, m->stream));
    }
    if (st == AK_OK) st = set_ctl(m, pos, 0, 0, 0);
    Prof p;
    mark(&p, m->stream, -1);
    if (st == AK_OK) st = emit_target(m, pf ? m->pf : m->dec, T, pf ? 0 : 1, &p);
    if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
    for (size_t i = 1; i < p.ev.size() && st == AK_OK; ++i) {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, p.ev[i - 1], p.ev[i]);
        ms_by_kind[p.kind[i]] += ms;
        count_by_kind[p.kind[i]] += 1;
    }
    for (auto e : p.ev) cudaEventDestroy(e);
    return st;
}
