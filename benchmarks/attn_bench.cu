// Standalone check and timing of the full-attention kernels on random data, no model needed (~150 MB of VRAM).
//
//   scripts\attn_bench.ps1        (builds build-dev\attn_bench.exe and runs it)
//
// It includes qwen35_model.cu so it calls the same (file-local) kernels the runtime launches, with the model's
// shapes: 4 KV heads x 4 query heads each, head dim 256, max_seq 40960. For each case it runs k_attn_mma (v1) and
// k_attn_mma2 (v2) on identical inputs, reports whether the outputs are bit-identical, the max |difference|, and the
// median time of 20 launches. Prefill cases (T = 512 / 1024 / 64 columns) check the gated fp16 output; decode /
// verify cases (T = 1 / 12) check the unnormalized split partials that k_attn_combine merges.

#include "qwen35_model.cu"

#include <random>

// The model code references these (defined in other files of the DLL); the bench never calls them.
extern "C" void ak_set_error(const char* msg) { std::fprintf(stderr, "ak error: %s\n", msg); }
extern "C" ak_status ak_init(int) { return AK_ERR_INVALID; }
ak_status ak_launch_task(const ak_task*, cudaStream_t) { return AK_ERR_INVALID; }

namespace {

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

constexpr int kHkvB = 4, kNrB = 4, kHqB = kHkvB * kNrB, kSeqB = 40960;
const int kNSplitB = split_count(kSeqB);

template <typename F>
float time_ms(F&& f, int reps = 20) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> t;
    f();  // warm-up
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return t[t.size() / 2];
}

struct Cmp {
    bool same;
    double max_diff;
};

Cmp compare_h(const __half* a, const __half* b, size_t n) {
    std::vector<__half> x(n), y(n);
    CK(cudaMemcpy(x.data(), a, n * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, n * sizeof(__half), cudaMemcpyDeviceToHost));
    Cmp c{std::memcmp(x.data(), y.data(), n * sizeof(__half)) == 0, 0.0};
    for (size_t i = 0; i < n; ++i) {
        c.max_diff = std::max(c.max_diff, std::fabs(static_cast<double>(__half2float(x[i])) - __half2float(y[i])));
    }
    return c;
}

Cmp compare_f(const float* a, const float* b, size_t n) {
    std::vector<float> x(n), y(n);
    CK(cudaMemcpy(x.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    Cmp c{std::memcmp(x.data(), y.data(), n * sizeof(float)) == 0, 0.0};
    for (size_t i = 0; i < n; ++i) c.max_diff = std::max(c.max_diff, std::fabs(static_cast<double>(x[i]) - y[i]));
    return c;
}

// fp16 cache -> int8 + scales via the runtime's kv8_store (one warp per head vector), and the reference dequantize
// (half(q) * scale, the same single fp16 rounding as attn_tile8_cvt).
__global__ void k_bench_kv8_quant(const __half* src, int8_t* q, __half* sc, size_t n_vec) {
    const size_t v = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (v >= n_vec) return;
    const int lane = threadIdx.x & 31;
    for (int j = 0; j < 8; ++j) kv8_store(__half2float(src[v * kHd + lane + 32 * j]), q, sc, v * kHd, j);
}
__global__ void k_bench_kv8_deq(const int8_t* q, const __half* sc, __half* out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hmul(__int2half_rn(q[i]), sc[i / 32]);
}

}  // namespace

int main(int argc, char** argv) {
    const int only = argc > 1 ? std::atoi(argv[1]) : -1;  // optional: run only case #only
    for (const void* f : {reinterpret_cast<const void*>(k_attn_mma<4, false>),
                          reinterpret_cast<const void*>(k_attn_mma<4, true>),
                          reinterpret_cast<const void*>(k_attn_mma2<4, false>),
                          reinterpret_cast<const void*>(k_attn_mma2<4, true>)}) {
        CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kAttnSmem)));
    }
    for (const void* f : {reinterpret_cast<const void*>(k_attn_mma2<4, false, true>),
                          reinterpret_cast<const void*>(k_attn_mma2<4, true, true>)}) {
        CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kAttnSmemKv8)));
    }
    const int max_t = 1024;
    const int qk_stride = kHqB * 2 * kHd;
    const size_t kv_n = static_cast<size_t>(kSeqB) * kHkvB * kHd;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    // Keys / values roughly like post-RMSNorm activations; q scaled so logits spread over a few units.
    std::vector<__half> hk(kv_n), hv(kv_n);
    for (size_t i = 0; i < kv_n; ++i) {
        hk[i] = __float2half(nd(rng));
        hv[i] = __float2half(nd(rng));
    }
    std::vector<float> hq(static_cast<size_t>(max_t) * kHqB * kHd), hqk(static_cast<size_t>(max_t) * qk_stride);
    for (auto& x : hq) x = 0.35f * nd(rng);
    for (auto& x : hqk) x = nd(rng);

    __half *kc, *vc, *o1, *o2;
    float *q, *qk, *p1, *p2;
    int* pos_d;
    const size_t o_n = static_cast<size_t>(max_t) * kHqB * kHd;
    const size_t part_n = static_cast<size_t>(kMaxCols) * kHqB * kNSplitB * (kHd + 2);
    CK(cudaMalloc(&kc, kv_n * sizeof(__half)));
    CK(cudaMalloc(&vc, kv_n * sizeof(__half)));
    CK(cudaMalloc(&q, hq.size() * sizeof(float)));
    CK(cudaMalloc(&qk, hqk.size() * sizeof(float)));
    CK(cudaMalloc(&o1, o_n * sizeof(__half)));
    CK(cudaMalloc(&o2, o_n * sizeof(__half)));
    CK(cudaMalloc(&p1, part_n * sizeof(float)));
    CK(cudaMalloc(&p2, part_n * sizeof(float)));
    CK(cudaMalloc(&pos_d, sizeof(int)));
    CK(cudaMemcpy(kc, hk.data(), kv_n * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(vc, hv.data(), kv_n * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(q, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(qk, hqk.data(), hqk.size() * sizeof(float), cudaMemcpyHostToDevice));

    struct Case {
        int T, pos;
    };
    const Case cases[] = {{512, 0}, {512, 4000}, {512, 16000}, {512, 32000}, {1024, 32000}, {64, 32000},
                          {1, 4000}, {1, 32000}, {12, 4000}, {12, 32000}};
    std::printf("%-6s %-6s | %-9s %-9s %-7s | %-6s %s\n", "T", "pos", "v1 ms", "v2 ms", "speedup", "same", "max|diff|");
    int idx = 0;
    bool all_same = true;
    for (const Case& cs : cases) {
        if (only >= 0 && idx++ != only) continue;
        CK(cudaMemcpy(pos_d, &cs.pos, sizeof(int), cudaMemcpyHostToDevice));
        const bool pf = cs.T > kMaxCols;
        const int tiles = (cs.T * kNrB + kAttnRows - 1) / kAttnRows;
        const dim3 grid = pf ? dim3(kHkvB, tiles) : dim3(kHkvB, kNSplitB, tiles);
        auto run = [&](int v) {
            if (pf) {
                if (v == 1)
                    k_attn_mma<4, false><<<grid, 128, kAttnSmem>>>(q, kc, vc, qk, qk_stride, o1, nullptr, pos_d, kHkvB,
                                                                   cs.T, kNSplitB, kSeqB);
                else
                    k_attn_mma2<4, false><<<grid, 128, kAttnSmem>>>(q, kc, vc, qk, qk_stride, o2, nullptr, pos_d, kHkvB,
                                                                    cs.T, kNSplitB, kSeqB);
            } else {
                if (v == 1)
                    k_attn_mma<4, true><<<grid, 128, kAttnSmem>>>(q, kc, vc, qk, qk_stride, nullptr, p1, pos_d, kHkvB,
                                                                  cs.T, kNSplitB, kSeqB);
                else
                    k_attn_mma2<4, true><<<grid, 128, kAttnSmem>>>(q, kc, vc, qk, qk_stride, nullptr, p2, pos_d, kHkvB,
                                                                   cs.T, kNSplitB, kSeqB);
            }
            CK(cudaGetLastError());
        };
        CK(cudaMemset(p1, 0, part_n * sizeof(float)));
        CK(cudaMemset(p2, 0, part_n * sizeof(float)));
        run(1);
        run(2);
        CK(cudaDeviceSynchronize());
        const Cmp c = pf ? compare_h(o1, o2, static_cast<size_t>(cs.T) * kHqB * kHd)
                         : compare_f(p1, p2, static_cast<size_t>(cs.T) * kHqB * kNSplitB * (kHd + 2));
        all_same = all_same && c.same;
        const float t1 = time_ms([&] { run(1); });
        const float t2 = time_ms([&] { run(2); });
        std::printf("%-6d %-6d | %-9.3f %-9.3f %-7.2f | %-6s %.3g\n", cs.T, cs.pos, t1, t2, t1 / t2,
                    c.same ? "yes" : "NO", c.max_diff);
    }
    std::printf(all_same ? "all outputs bit-identical\n" : "OUTPUTS DIFFER\n");

    // ---- 8-bit KV cache: v2 KV8 vs v2 fp16 on the dequantized cache (must be bit-identical), and vs fp16 v2 on
    // the original cache (the quantization error), plus timing.
    int8_t *kq, *vq;
    __half *ksc, *vsc, *kd, *vd;
    const size_t n_vec = kv_n / kHd;
    CK(cudaMalloc(&kq, kv_n));
    CK(cudaMalloc(&vq, kv_n));
    CK(cudaMalloc(&ksc, n_vec * 8 * sizeof(__half)));
    CK(cudaMalloc(&vsc, n_vec * 8 * sizeof(__half)));
    CK(cudaMalloc(&kd, kv_n * sizeof(__half)));
    CK(cudaMalloc(&vd, kv_n * sizeof(__half)));
    const unsigned qb = static_cast<unsigned>((n_vec * 32 + 255) / 256), db = static_cast<unsigned>((kv_n + 255) / 256);
    k_bench_kv8_quant<<<qb, 256>>>(kc, kq, ksc, n_vec);
    k_bench_kv8_quant<<<qb, 256>>>(vc, vq, vsc, n_vec);
    k_bench_kv8_deq<<<db, 256>>>(kq, ksc, kd, kv_n);
    k_bench_kv8_deq<<<db, 256>>>(vq, vsc, vd, kv_n);
    CK(cudaGetLastError());
    __half* o3;
    float* p3;
    CK(cudaMalloc(&o3, o_n * sizeof(__half)));
    CK(cudaMalloc(&p3, part_n * sizeof(float)));
    std::printf("\nKV8:  %-6s %-6s | %-9s %-9s %-7s | %-6s %-9s | %s\n", "T", "pos", "fp16 ms", "kv8 ms", "speedup",
                "same", "max|diff|", "vs original fp16 cache: max|diff|");
    bool kv8_same = true;
    idx = 0;
    for (const Case& cs : cases) {
        if (only >= 0 && idx++ != only) continue;
        CK(cudaMemcpy(pos_d, &cs.pos, sizeof(int), cudaMemcpyHostToDevice));
        const bool pf = cs.T > kMaxCols;
        const int tiles = (cs.T * kNrB + kAttnRows - 1) / kAttnRows;
        const dim3 grid = pf ? dim3(kHkvB, tiles) : dim3(kHkvB, kNSplitB, tiles);
        // mode 0: fp16 v2 on the original cache, 1: fp16 v2 on the dequantized cache, 2: KV8
        auto run = [&](int mode) {
            __half* o = mode == 0 ? o1 : mode == 1 ? o2 : o3;
            float* pp = mode == 0 ? p1 : mode == 1 ? p2 : p3;
            const void* K = mode == 0 ? static_cast<const void*>(kc) : mode == 1 ? static_cast<const void*>(kd) : kq;
            const void* V = mode == 0 ? static_cast<const void*>(vc) : mode == 1 ? static_cast<const void*>(vd) : vq;
            if (pf) {
                if (mode == 2)
                    k_attn_mma2<4, false, true><<<grid, 128, kAttnSmemKv8>>>(q, K, V, qk, qk_stride, o, nullptr, pos_d,
                                                                           kHkvB, cs.T, kNSplitB, kSeqB, ksc, vsc);
                else
                    k_attn_mma2<4, false><<<grid, 128, kAttnSmem>>>(q, K, V, qk, qk_stride, o, nullptr, pos_d, kHkvB,
                                                                    cs.T, kNSplitB, kSeqB);
            } else {
                if (mode == 2)
                    k_attn_mma2<4, true, true><<<grid, 128, kAttnSmemKv8>>>(q, K, V, qk, qk_stride, nullptr, pp, pos_d,
                                                                          kHkvB, cs.T, kNSplitB, kSeqB, ksc, vsc);
                else
                    k_attn_mma2<4, true><<<grid, 128, kAttnSmem>>>(q, K, V, qk, qk_stride, nullptr, pp, pos_d, kHkvB,
                                                                   cs.T, kNSplitB, kSeqB);
            }
            CK(cudaGetLastError());
        };
        CK(cudaMemset(p1, 0, part_n * sizeof(float)));
        CK(cudaMemset(p2, 0, part_n * sizeof(float)));
        CK(cudaMemset(p3, 0, part_n * sizeof(float)));
        run(0);
        run(1);
        run(2);
        CK(cudaDeviceSynchronize());
        const size_t n = pf ? static_cast<size_t>(cs.T) * kHqB * kHd
                            : static_cast<size_t>(cs.T) * kHqB * kNSplitB * (kHd + 2);
        const Cmp c = pf ? compare_h(o2, o3, n) : compare_f(p2, p3, n);
        const Cmp e = pf ? compare_h(o1, o3, n) : compare_f(p1, p3, n);
        kv8_same = kv8_same && c.same;
        const float t0 = time_ms([&] { run(0); });
        const float t2 = time_ms([&] { run(2); });
        std::printf("      %-6d %-6d | %-9.3f %-9.3f %-7.2f | %-6s %-9.3g | %.3g\n", cs.T, cs.pos, t0, t2, t0 / t2,
                    c.same ? "yes" : "NO", c.max_diff, e.max_diff);
    }
    std::printf(kv8_same ? "KV8 bit-identical to fp16 attention on the dequantized cache\n" : "KV8 OUTPUTS DIFFER\n");
    all_same = all_same && kv8_same;
    return all_same ? 0 : 2;
}
