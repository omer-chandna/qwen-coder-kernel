// Standalone check and timing of the prefill int8 GEMMs (k_mmq4 / k_mmq6) on random quantized weights at the model's
// shapes, no model load.
//
//   scripts\gemm_bench.ps1 [-Args "..."]
//
// Includes qwen35_model.cu, so the kernels and their launch configurations are the runtime's own. For each matrix
// (rows x K, Q4_K or Q6_K) and token count T it reports:
//   - median ms of 10 launches and the MAC rate (rows * K * T / time);
//   - the worst relative error over sampled outputs against a CPU fp64 reference that uses the same Q8_1 activations
//     (int8 per 32 values), so it measures the kernel's own arithmetic, not activation quantization;
//   - with DIAG kernels: time with the per-sub-block float epilogue reduced to one add per output (DIAG 1) and,
//     for Q4_K, without the min term (DIAG 2). Their results are wrong by design; they show what the epilogue costs.

#include "qwen35_model.cu"

#include <random>

extern "C" void ak_set_error(const char* msg) { std::fprintf(stderr, "ak error: %s\n", msg); }
extern "C" ak_status ak_init(int) { return AK_ERR_INVALID; }
ak_status ak_launch_task(const ak_task*, cudaStream_t) { return AK_ERR_INVALID; }

namespace {

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

template <typename F>
float time_ms(F&& f, int reps = 10) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> t;
    f();
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

float h2f(uint16_t h) {
    __half x;
    std::memcpy(&x, &h, 2);
    return __half2float(x);
}
uint16_t f2h(float f) {
    const __half x = __float2half(f);
    uint16_t h;
    std::memcpy(&h, &x, 2);
    return h;
}

// Random but well-formed super-blocks: random quant bytes and scales, small positive d / dmin.
std::vector<uint8_t> make_weights(int type, int rows, int K, std::mt19937& rng) {
    const int bb = type == 1 ? 144 : 210;
    std::vector<uint8_t> w(static_cast<size_t>(rows) * (K / 256) * bb);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    for (auto& x : w) x = static_cast<uint8_t>(byte(rng));
    for (size_t b = 0; b < w.size() / bb; ++b) {
        uint8_t* blk = w.data() + b * bb;
        if (type == 1) {
            const uint16_t d = f2h(0.002f * ud(rng)), dm = f2h(0.004f * ud(rng));
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dm, 2);
        } else {
            const uint16_t d = f2h(0.0008f * ud(rng));
            std::memcpy(blk + 208, &d, 2);
        }
    }
    return w;
}

// Dequantized weight row (fp64), llama.cpp's Q4_K / Q6_K layouts.
void dequant_row(int type, const uint8_t* row, int K, std::vector<double>& out) {
    out.assign(K, 0.0);
    for (int b = 0; b < K / 256; ++b) {
        double* y = out.data() + b * 256;
        if (type == 1) {
            const uint8_t* blk = row + b * 144;
            uint16_t dh, mh;
            std::memcpy(&dh, blk, 2);
            std::memcpy(&mh, blk + 2, 2);
            const double d = h2f(dh), dmin = h2f(mh);
            const uint8_t* q = blk + 4;
            const uint8_t* qs = blk + 16;
            for (int j = 0; j < 8; ++j) {
                int sc, mn;
                if (j < 4) {
                    sc = q[j] & 63;
                    mn = q[j + 4] & 63;
                } else {
                    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
                }
                for (int l = 0; l < 32; ++l) {
                    const uint8_t byte = qs[(j / 2) * 32 + l];
                    const int v = (j & 1) ? byte >> 4 : byte & 0xF;
                    y[j * 32 + l] = d * sc * v - dmin * mn;
                }
            }
        } else {
            const uint8_t* blk = row + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
            uint16_t dh;
            std::memcpy(&dh, blk + 208, 2);
            const double d = h2f(dh);
            for (int n = 0; n < 2; ++n) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[n * 128 + l] = d * sc[is] * q1;
                    y[n * 128 + l + 32] = d * sc[is + 2] * q2;
                    y[n * 128 + l + 64] = d * sc[is + 4] * q3;
                    y[n * 128 + l + 96] = d * sc[is + 6] * q4;
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }
}

template <int DIAG>
void launch(int type, int T, const uint8_t* W, int rows, int K, const int8_t* xq, const float* xd, const float* xsd,
            float* Y, int nsplit = 1) {
    using namespace akd::qgemm;
    auto go = [&](auto bn_tag, auto s_tag) {
        constexpr int BN = decltype(bn_tag)::value, S = decltype(s_tag)::value;
        const dim3 g((T + BN - 1) / BN, (rows + BM - 1) / BM, nsplit);
        if (type == 1) {
            static bool init = false;
            if (!init) {
                CK(cudaFuncSetAttribute(reinterpret_cast<const void*>(k_mmq4<BN, S, 1, DIAG>),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(p4_smem_bytes<BN, S>())));
                init = true;
            }
            k_mmq4<BN, S, 1, DIAG><<<g, NT, p4_smem_bytes<BN, S>()>>>(W, xq, xd, xsd, Y, rows, K, T, rows, nsplit);
        } else {
            static bool init = false;
            if (!init) {
                CK(cudaFuncSetAttribute(reinterpret_cast<const void*>(k_mmq6<BN, S, 1, DIAG>),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(p6_smem_bytes<BN, S>())));
                init = true;
            }
            k_mmq6<BN, S, 1, DIAG><<<g, NT, p6_smem_bytes<BN, S>()>>>(W, xq, xd, xsd, Y, rows, K, T, rows, nsplit);
        }
    };
    // The runtime's default configurations: cfg 0 (BN 64, 3 stages) up to 64 tokens, cfg 2 (BN 128, 2 stages) above.
    if (T <= 64) go(std::integral_constant<int, 64>{}, std::integral_constant<int, 3>{});
    else go(std::integral_constant<int, 128>{}, std::integral_constant<int, 2>{});
    CK(cudaGetLastError());
}

}  // namespace

int main() {
    struct Mat {
        const char* name;
        int type, rows, K;
    };
    const Mat mats[] = {{"ffn_gate Q4_K", 1, 12288, 4096}, {"ffn_down Q4_K", 1, 4096, 12288},
                        {"ffn_down Q6_K", 2, 4096, 12288}, {"ssm qkv  Q6_K", 2, 8192, 4096},
                        {"ssm_out  Q4_K", 1, 4096, 4096}};
    const int Ts[] = {64, 128, 256, 512};
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::printf("%-14s %5s | %8s %7s | %9s | %8s %7s | %8s %7s\n", "matrix", "T", "ms", "TMAC/s", "max rel", "diag1 ms",
                "TMAC/s", "diag2 ms", "TMAC/s");
    for (const Mat& M : mats) {
        const std::vector<uint8_t> hw = make_weights(M.type, M.rows, M.K, rng);
        uint8_t* W;
        CK(cudaMalloc(&W, hw.size() + 16));
        CK(cudaMemcpy(W, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        for (int T : Ts) {
            // fp16 activations with a few large channels, like post-norm hidden states.
            std::vector<__half> hx(static_cast<size_t>(T) * M.K);
            for (size_t i = 0; i < hx.size(); ++i) {
                const float big = (i % M.K) % 97 == 0 ? 20.f : 1.f;
                hx[i] = __float2half(big * nd(rng));
            }
            __half* X;
            int8_t* xq;
            float *xd, *xsd, *Y, *P, *Ys;
            CK(cudaMalloc(&X, hx.size() * 2));
            CK(cudaMalloc(&xq, hx.size()));
            CK(cudaMalloc(&xd, hx.size() / 32 * 4));
            CK(cudaMalloc(&xsd, hx.size() / 32 * 4));
            CK(cudaMalloc(&Y, static_cast<size_t>(T) * M.rows * 4));
            CK(cudaMalloc(&Ys, static_cast<size_t>(T) * M.rows * 4));
            CK(cudaMalloc(&P, static_cast<size_t>(4) * T * M.rows * 4));
            CK(cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            akd::qgemm::k_quant_x8<<<dim3(M.K / 256, T), 256>>>(X, M.K, xq, xd, xsd);
            CK(cudaGetLastError());

            launch<0>(M.type, T, W, M.rows, M.K, xq, xd, xsd, Y);
            CK(cudaDeviceSynchronize());
            std::vector<float> hy(static_cast<size_t>(T) * M.rows);
            std::vector<int8_t> hq(hx.size());
            std::vector<float> hd(hx.size() / 32);
            CK(cudaMemcpy(hy.data(), Y, hy.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hq.data(), xq, hq.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hd.data(), xd, hd.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0.0;
            std::vector<double> wr;
            const size_t row_bytes = static_cast<size_t>(M.K / 256) * (M.type == 1 ? 144 : 210);
            for (int r = 0; r < M.rows; r += 509) {
                dequant_row(M.type, hw.data() + r * row_bytes, M.K, wr);
                for (int t = 0; t < T; t += 37) {
                    double ref = 0.0, mag = 0.0;
                    for (int k = 0; k < M.K; ++k) {
                        const double xv = hq[static_cast<size_t>(t) * M.K + k] * double(hd[(static_cast<size_t>(t) * M.K + k) / 32]);
                        ref += wr[k] * xv;
                        mag += std::fabs(wr[k] * xv);
                    }
                    worst = std::max(worst, std::fabs(hy[static_cast<size_t>(t) * M.rows + r] - ref) / mag);
                }
            }
            const double macs = double(M.rows) * M.K * T;
            const float t0 = time_ms([&] { launch<0>(M.type, T, W, M.rows, M.K, xq, xd, xsd, Y); });
            const float t1 = time_ms([&] { launch<1>(M.type, T, W, M.rows, M.K, xq, xd, xsd, Y); });
            float t2 = 0.f, t3 = 0.f;
            if (M.type == 1) t2 = time_ms([&] { launch<2>(M.type, T, W, M.rows, M.K, xq, xd, xsd, Y); });
            if (M.type == 1) {
                t3 = time_ms([&] { launch<4>(M.type, T, W, M.rows, M.K, xq, xd, xsd, Y); });
                const float t5 = time_ms([&] { launch<5>(M.type, T, W, M.rows, M.K, xq, xd, xsd, Y); });
                const float t6 = time_ms([&] { launch<6>(M.type, T, W, M.rows, M.K, xq, xd, xsd, Y); });
                std::printf("   no mma (DIAG 4): %.3f ms | no copies (DIAG 5): %.3f ms | no scale decode (DIAG 6): %.3f ms\n",
                            t3, t5, t6);
            }
            std::printf("%-14s %5d | %8.3f %7.1f | %9.2e | %8.3f %7.1f | %8.3f %7.1f\n", M.name, T, t0,
                        macs / t0 * 1e-9, worst, t1, macs / t1 * 1e-9, t2, t2 > 0 ? macs / t2 * 1e-9 : 0.0);
            // split-K: kernel into planes of P, then the fixed-order reduction
            for (int z = 2; z <= 4; ++z) {
                if ((M.K / 256) % z != 0 || M.K / 256 / z < 4) continue;
                auto run = [&] {
                    launch<0>(M.type, T, W, M.rows, M.K, xq, xd, xsd, P, z);
                    akd::qgemm::k_splitk_sum<<<dim3((M.rows + 255) / 256, T), 256>>>(P, z, M.rows, T, Ys, M.rows);
                };
                run();
                CK(cudaDeviceSynchronize());
                std::vector<float> ys(hy.size());
                CK(cudaMemcpy(ys.data(), Ys, ys.size() * 4, cudaMemcpyDeviceToHost));
                double d = 0.0, mx = 0.0;
                for (size_t i = 0; i < ys.size(); ++i) {
                    d = std::max(d, static_cast<double>(std::fabs(ys[i] - hy[i])));
                    mx = std::max(mx, static_cast<double>(std::fabs(hy[i])));
                }
                const float ts = time_ms(run);
                std::printf("   split-K %d: %.3f ms (%.2fx), max |diff| / max |y| = %.1e\n", z, ts, t0 / ts, d / mx);
            }
            CK(cudaFree(X));
            CK(cudaFree(xq));
            CK(cudaFree(xd));
            CK(cudaFree(xsd));
            CK(cudaFree(Y));
            CK(cudaFree(Ys));
            CK(cudaFree(P));
        }
        CK(cudaFree(W));
    }
    return 0;
}
