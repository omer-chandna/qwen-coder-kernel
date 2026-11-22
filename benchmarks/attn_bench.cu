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
constexpr int kNSplitB = (kSeqB + kAttnChunk - 1) / kAttnChunk;

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

}  // namespace

int main(int argc, char** argv) {
    const int only = argc > 1 ? std::atoi(argv[1]) : -1;  // optional: run only case #only
    for (const void* f : {reinterpret_cast<const void*>(k_attn_mma<4, false>),
                          reinterpret_cast<const void*>(k_attn_mma<4, true>),
                          reinterpret_cast<const void*>(k_attn_mma2<4, false>),
                          reinterpret_cast<const void*>(k_attn_mma2<4, true>)}) {
        CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kAttnSmem)));
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
                (v == 1 ? k_attn_mma<4, false> : k_attn_mma2<4, false>)<<<grid, 128, kAttnSmem>>>(
                    q, kc, vc, qk, qk_stride, v == 1 ? o1 : o2, nullptr, pos_d, kHkvB, cs.T, kNSplitB, kSeqB);
            } else {
                (v == 1 ? k_attn_mma<4, true> : k_attn_mma2<4, true>)<<<grid, 128, kAttnSmem>>>(
                    q, kc, vc, qk, qk_stride, nullptr, v == 1 ? p1 : p2, pos_d, kHkvB, cs.T, kNSplitB, kSeqB);
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
    return all_same ? 0 : 2;
}
