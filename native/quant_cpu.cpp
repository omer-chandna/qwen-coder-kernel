// Q8_0 / Q4_K / Q6_K block layout follows llama.cpp ggml (MIT license).
// The Q4_K dequantization and 6-bit scale packing match ggml-quants.c.
// The Q4_K quantizer is a direct min/max quantizer into that layout, not llama.cpp's
// iterative rate-distortion search.

#include <adakernel/api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

char g_error[1024] = "";

void set_error(const char* msg) {
    std::snprintf(g_error, sizeof(g_error), "%s", msg ? msg : "");
}

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t mant = x & 0x007fffffu;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>(sign | 0x7e00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t m = mant | 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(1 - exp + 13);
        uint32_t t = m >> shift;
        const uint32_t rem_mask = (shift >= 32) ? 0xffffffffu : ((1u << shift) - 1u);
        const uint32_t rem = m & rem_mask;
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (t & 1u))) {
            ++t;
        }
        return static_cast<uint16_t>(sign | t);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t half_m = mant >> 13;
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u))) {
        ++half_m;
        if (half_m == 0x400u) {
            half_m = 0;
            ++exp;
            if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | half_m);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h & 0x7c00u) >> 10;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 127 - 14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

#pragma pack(push, 1)
struct BlockQ8 {
    uint16_t d;
    int8_t qs[32];
};
struct BlockQ4 {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct BlockQ6 {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ8) == 34, "q8_0 size");
static_assert(sizeof(BlockQ4) == 144, "q4_K size");
static_assert(sizeof(BlockQ6) == 210, "q6_K size");

}  // namespace

extern "C" const char* ak_last_error(void) { return g_error; }

extern "C" void ak_set_error(const char* msg) { set_error(msg); }

extern "C" int ak_q8_block_size(void) { return 34; }
extern "C" int ak_q4k_block_size(void) { return 144; }
extern "C" int ak_q6k_block_size(void) { return 210; }

extern "C" ak_status ak_q8_quantize(const float* x, void* dst, int n) {
    if (!x || !dst || n <= 0 || (n % 32) != 0) {
        set_error("q8 quantize: n must be a positive multiple of 32");
        return AK_ERR_INVALID;
    }
    auto* y = static_cast<BlockQ8*>(dst);
    const int nb = n / 32;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(x[i * 32 + j]));
        }
        const float d = amax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        y[i].d = f32_to_f16(d);
        for (int j = 0; j < 32; ++j) {
            int q = static_cast<int>(std::round(x[i * 32 + j] * id));
            q = std::max(-127, std::min(127, q));
            y[i].qs[j] = static_cast<int8_t>(q);
        }
    }
    return AK_OK;
}

extern "C" ak_status ak_q8_dequantize(const void* src, float* y, int n) {
    if (!src || !y || n <= 0 || (n % 32) != 0) {
        set_error("q8 dequantize: bad size");
        return AK_ERR_INVALID;
    }
    const auto* x = static_cast<const BlockQ8*>(src);
    const int nb = n / 32;
    for (int i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        for (int j = 0; j < 32; ++j) {
            y[i * 32 + j] = static_cast<float>(x[i].qs[j]) * d;
        }
    }
    return AK_OK;
}

extern "C" ak_status ak_q4k_quantize(const float* x, void* dst, int n) {
    if (!x || !dst || n <= 0 || (n % 256) != 0) {
        set_error("q4_K quantize: n must be a positive multiple of 256");
        return AK_ERR_INVALID;
    }
    auto* y = static_cast<BlockQ4*>(dst);
    const int nb = n / 256;
    for (int bi = 0; bi < nb; ++bi) {
        const float* row = x + bi * 256;
        BlockQ4 blk{};
        float scales[8];
        float mins[8];
        uint8_t L[256];
        for (int g = 0; g < 8; ++g) {
            float vmin = row[g * 32];
            float vmax = vmin;
            for (int l = 1; l < 32; ++l) {
                const float v = row[g * 32 + l];
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
            float d = (vmax - vmin) / 15.f;
            float m = -vmin;
            if (d < 0.f) d = 0.f;
            if (m < 0.f) m = 0.f;
            scales[g] = d;
            mins[g] = m;
        }
        float max_d = 0.f;
        float max_m = 0.f;
        for (int g = 0; g < 8; ++g) {
            max_d = std::max(max_d, scales[g]);
            max_m = std::max(max_m, mins[g]);
        }
        const float d_super = max_d > 0.f ? max_d / 63.f : 0.f;
        const float m_super = max_m > 0.f ? max_m / 63.f : 0.f;
        uint8_t sc[8], mn[8];
        for (int g = 0; g < 8; ++g) {
            int ls = d_super > 0.f ? static_cast<int>(std::round(scales[g] / d_super)) : 0;
            int lm = m_super > 0.f ? static_cast<int>(std::round(mins[g] / m_super)) : 0;
            sc[g] = static_cast<uint8_t>(std::max(0, std::min(63, ls)));
            mn[g] = static_cast<uint8_t>(std::max(0, std::min(63, lm)));
        }
        for (int g = 0; g < 8; ++g) {
            if (g < 4) {
                blk.scales[g] = sc[g];
                blk.scales[g + 4] = mn[g];
            } else {
                blk.scales[g + 4] = static_cast<uint8_t>((sc[g] & 0xF) | ((mn[g] & 0xF) << 4));
                blk.scales[g - 4] = static_cast<uint8_t>(blk.scales[g - 4] | ((sc[g] >> 4) << 6));
                blk.scales[g - 0] = static_cast<uint8_t>(blk.scales[g - 0] | ((mn[g] >> 4) << 6));
            }
        }
        blk.d = f32_to_f16(d_super);
        blk.dmin = f32_to_f16(m_super);
        const float d_rt = f16_to_f32(blk.d);
        const float m_rt = f16_to_f32(blk.dmin);
        for (int g = 0; g < 8; ++g) {
            uint8_t s, m;
            get_scale_min_k4(g, blk.scales, &s, &m);
            const float dd = d_rt * static_cast<float>(s);
            const float mm = m_rt * static_cast<float>(m);
            for (int l = 0; l < 32; ++l) {
                int q = 0;
                if (dd != 0.f) {
                    q = static_cast<int>(std::round((row[g * 32 + l] + mm) / dd));
                }
                q = std::max(0, std::min(15, q));
                L[g * 32 + l] = static_cast<uint8_t>(q);
            }
        }
        uint8_t* q = blk.qs;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = static_cast<uint8_t>(L[j + l] | (L[j + l + 32] << 4));
            }
            q += 32;
        }
        y[bi] = blk;
    }
    return AK_OK;
}

extern "C" ak_status ak_q4k_dequantize(const void* src, float* y, int n) {
    if (!src || !y || n <= 0 || (n % 256) != 0) {
        set_error("q4_K dequantize: bad size");
        return AK_ERR_INVALID;
    }
    const auto* x = static_cast<const BlockQ4*>(src);
    const int nb = n / 256;
    float* out = y;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d = f16_to_f32(x[i].d);
        const float minv = f16_to_f32(x[i].dmin);
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = minv * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = minv * m;
            for (int l = 0; l < 32; ++l) {
                *out++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *out++ = d2 * static_cast<float>(q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
    return AK_OK;
}

extern "C" ak_status ak_q6k_dequantize(const void* src, float* y, int n) {
    if (!src || !y || n <= 0 || (n % 256) != 0) {
        set_error("q6_K dequantize: bad size");
        return AK_ERR_INVALID;
    }
    const auto* x = static_cast<const BlockQ6*>(src);
    const int nb = n / 256;
    for (int i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        float* out = y + i * 256;
        for (int n64 = 0; n64 < 256; n64 += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l] = d * static_cast<float>(sc[is]) * static_cast<float>(q1);
                out[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                out[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                out[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return AK_OK;
}
