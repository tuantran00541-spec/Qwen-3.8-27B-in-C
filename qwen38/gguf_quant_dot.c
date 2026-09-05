/* Portable scalar quantized dot/runtime bridges for the Qwen3.8 lab.
 * Semantics mirror llama.cpp pin 557614e0296ff4a5b6f649737a65ae2076eea2fd.
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_QK8_0 32
#define QWEN_BLOCK_Q8_0 34
#define QWEN_QK_K 256
#define QWEN_BLOCK_Q6_K 210
#define QWEN_BLOCK_Q8_K 292

static float qwen_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                ++shift;
            }
            mant &= 0x03ffu;
            /* Half subnormals have an unbiased exponent of -14 before the
             * mantissa normalization shift. Using -15 here halves every
             * subnormal value, which real Q6_K super-scales can expose. */
            const uint32_t fexp = (uint32_t)(127 - 14 - shift);
            bits = sign | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        const uint32_t fexp = exp + (127u - 15u);
        bits = sign | (fexp << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* IEEE-754 binary32 -> binary16, round-to-nearest-even.  Keeping this local
 * avoids depending on compiler-specific _Float16 support in the future
 * Windows CPU runtime while matching the FP16 scale storage used by GGML. */
static uint16_t qwen_f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp32 = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x007fffffu;

    if (exp32 == 0xffu) {
        if (mant == 0) return (uint16_t)(sign | 0x7c00u);
        uint16_t payload = (uint16_t)(mant >> 13);
        if (payload == 0) payload = 1;
        return (uint16_t)(sign | 0x7c00u | payload | 0x0200u);
    }

    int exp16 = (int)exp32 - 127 + 15;
    if (exp16 >= 31) return (uint16_t)(sign | 0x7c00u);

    if (exp16 <= 0) {
        if (exp16 < -10) return sign;
        mant |= 0x00800000u;
        const int shift = 14 - exp16;
        uint32_t half_mant = mant >> shift;
        const uint32_t mask = (1u << shift) - 1u;
        const uint32_t remainder = mant & mask;
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
            ++half_mant;
        }
        return (uint16_t)(sign | half_mant);
    }

    uint32_t half_mant = mant >> 13;
    const uint32_t remainder = mant & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant == 0x0400u) {
            half_mant = 0;
            ++exp16;
            if (exp16 >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint16_t)exp16 << 10) | (uint16_t)half_mant);
}

/* Exact scalar helper used by llama.cpp's Q8_K reference quantizer.  The same
 * FP32 nearest-even primitive also matches the AVX/AVX2 Q8_0 activation
 * runtime path (_MM_ROUND_NEAREST).  This deliberately differs on exact .5
 * ties from quantize_row_q8_0_ref(), which is a deterministic file-creation
 * reference and uses roundf(). */
static int qwen_nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.0f);
    const float val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static int qwen_host_little_endian(void) {
    const uint16_t one = 1;
    return *((const uint8_t *)&one) == 1;
}

static uint16_t qwen_load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static void qwen_store_u16_le(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)(value & 0xffu);
    p[1] = (uint8_t)(value >> 8);
}

static float qwen_load_f32_le(const uint8_t *p) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void qwen_store_f32_le(uint8_t *p, float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    p[0] = (uint8_t)(bits & 0xffu);
    p[1] = (uint8_t)((bits >> 8) & 0xffu);
    p[2] = (uint8_t)((bits >> 16) & 0xffu);
    p[3] = (uint8_t)((bits >> 24) & 0xffu);
}

int qwen_quantize_q8_0_scalar(const float *x, size_t n, uint8_t *dst, size_t dst_bytes) {
    if (!x || !dst || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t nb = n / QWEN_QK8_0;
    if (dst_bytes != nb * QWEN_BLOCK_Q8_0) return -2;

    for (size_t ib = 0; ib < nb; ++ib) {
        const float *xb = x + ib * QWEN_QK8_0;
        uint8_t *out = dst + ib * QWEN_BLOCK_Q8_0;
        float amax = 0.0f;
        for (size_t j = 0; j < QWEN_QK8_0; ++j) {
            amax = fmaxf(amax, fabsf(xb[j]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        qwen_store_u16_le(out, qwen_f32_to_f16(d));
        int8_t *qs = (int8_t *)(out + 2);
        for (size_t j = 0; j < QWEN_QK8_0; ++j) {
            int q = qwen_nearest_int(xb[j] * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            qs[j] = (int8_t)q;
        }
    }
    return 0;
}

float qwen_vec_dot_q8_0_q8_0_scalar(const uint8_t *x, size_t x_bytes, const uint8_t *y, size_t y_bytes, size_t n) {
    if (!x || !y || n == 0 || n % QWEN_QK8_0 != 0) return NAN;
    const size_t nb = n / QWEN_QK8_0;
    if (x_bytes != nb * QWEN_BLOCK_Q8_0 || y_bytes != nb * QWEN_BLOCK_Q8_0) return NAN;

    float sum = 0.0f;
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = x + ib * QWEN_BLOCK_Q8_0;
        const uint8_t *yb = y + ib * QWEN_BLOCK_Q8_0;
        const int8_t *xq = (const int8_t *)(xb + 2);
        const int8_t *yq = (const int8_t *)(yb + 2);
        int32_t sumi = 0;
        for (size_t j = 0; j < QWEN_QK8_0; ++j) {
            sumi += (int32_t)xq[j] * (int32_t)yq[j];
        }
        const float dx = qwen_f16_to_f32(qwen_load_u16_le(xb));
        const float dy = qwen_f16_to_f32(qwen_load_u16_le(yb));
        sum += (float)sumi * dx * dy;
    }
    return sum;
}

int qwen_quantize_q8_k_scalar(const float *x, size_t n, uint8_t *dst, size_t dst_bytes) {
    if (!x || !dst || n == 0 || n % QWEN_QK_K != 0) return -1;
    const size_t nb = n / QWEN_QK_K;
    if (dst_bytes != nb * QWEN_BLOCK_Q8_K) return -2;

    for (size_t ib = 0; ib < nb; ++ib) {
        const float *xb = x + ib * QWEN_QK_K;
        uint8_t *out = dst + ib * QWEN_BLOCK_Q8_K;
        float amax = 0.0f;
        float maxv = 0.0f;
        for (size_t j = 0; j < QWEN_QK_K; ++j) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                maxv = xb[j];
            }
        }
        if (amax == 0.0f) {
            memset(out, 0, QWEN_BLOCK_Q8_K);
            continue;
        }
        const float iscale = -127.0f / maxv;
        const float d = 1.0f / iscale;
        qwen_store_f32_le(out, d);
        int8_t *qs = (int8_t *)(out + 4);
        for (size_t j = 0; j < QWEN_QK_K; ++j) {
            int q = qwen_nearest_int(iscale * xb[j]);
            if (q > 127) q = 127;
            qs[j] = (int8_t)q;
        }
        uint8_t *bs = out + 4 + QWEN_QK_K;
        for (size_t j = 0; j < QWEN_QK_K / 16; ++j) {
            int sum = 0;
            for (size_t ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            const int16_t s = (int16_t)sum;
            bs[2*j + 0] = (uint8_t)((uint16_t)s & 0xffu);
            bs[2*j + 1] = (uint8_t)(((uint16_t)s >> 8) & 0xffu);
        }
    }
    return 0;
}

float qwen_vec_dot_q6_k_q8_k_scalar(const uint8_t *q6, size_t q6_bytes, const uint8_t *q8k, size_t q8k_bytes, size_t n) {
    if (!q6 || !q8k || n == 0 || n % QWEN_QK_K != 0) return NAN;
    const size_t nb = n / QWEN_QK_K;
    if (q6_bytes != nb * QWEN_BLOCK_Q6_K || q8k_bytes != nb * QWEN_BLOCK_Q8_K) return NAN;

    float sums[8] = {0};
    int8_t aux8[QWEN_QK_K];
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = q6 + ib * QWEN_BLOCK_Q6_K;
        const uint8_t *yb = q8k + ib * QWEN_BLOCK_Q8_K;
        const uint8_t *ql = xb;
        const uint8_t *qh = xb + 128;
        const int8_t *scales = (const int8_t *)(xb + 192);
        const int8_t *q8 = (const int8_t *)(yb + 4);

        int8_t *a = aux8;
        for (int j = 0; j < QWEN_QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0]  = (int8_t)(((ql[l + 0]  & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = (int8_t)(((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = (int8_t)(((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = (int8_t)(((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            a += 128;
            ql += 64;
            qh += 32;
        }

        int32_t lanes[8] = {0};
        a = aux8;
        int is = 0;
        for (int j = 0; j < QWEN_QK_K/16; ++j) {
            const int scale = scales[is++];
            for (int l = 0; l < 8; ++l) lanes[l] += scale * (int)q8[l] * (int)a[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) lanes[l] += scale * (int)q8[l] * (int)a[l];
            q8 += 8; a += 8;
        }
        const float d = qwen_f16_to_f32(qwen_load_u16_le(xb + 208)) * qwen_load_f32_le(yb);
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)lanes[l];
    }
    float sumf = 0.0f;
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

/* Matrix storage follows GGML's contiguous row layout: each output row owns
 * all packed blocks for ne0. The activation is already quantized once and is
 * reused across every row, which is the shape needed by the K3 runtime. */
int qwen_matvec_q8_0_q8_0_scalar(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t row_bytes = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    if (activation_bytes != row_bytes) return -2;
    if (weights_bytes % row_bytes != 0 || weights_bytes / row_bytes != rows) return -3;
    for (size_t row = 0; row < rows; ++row) {
        out[row] = qwen_vec_dot_q8_0_q8_0_scalar(
            weights + row * row_bytes, row_bytes, activation, activation_bytes, n);
    }
    return 0;
}

int qwen_matvec_q6_k_q8_k_scalar(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n % QWEN_QK_K != 0) return -1;
    const size_t weight_row_bytes = (n / QWEN_QK_K) * QWEN_BLOCK_Q6_K;
    const size_t activation_row_bytes = (n / QWEN_QK_K) * QWEN_BLOCK_Q8_K;
    if (activation_bytes != activation_row_bytes) return -2;
    if (weights_bytes % weight_row_bytes != 0 || weights_bytes / weight_row_bytes != rows) return -3;
    for (size_t row = 0; row < rows; ++row) {
        out[row] = qwen_vec_dot_q6_k_q8_k_scalar(
            weights + row * weight_row_bytes, weight_row_bytes,
            activation, activation_bytes, n);
    }
    return 0;
}

#ifdef QWEN_QUANT_DOT_SELFTEST
static void pack_q6_fixture(uint8_t out[QWEN_BLOCK_Q6_K], uint16_t f16_scale, const int8_t scales[16], const int8_t quants[256]) {
    memset(out, 0, QWEN_BLOCK_Q6_K);
    uint8_t *ql = out;
    uint8_t *qh = out + 128;
    for (int n = 0; n < 256; n += 128) {
        const int ql_base = n == 0 ? 0 : 64;
        const int qh_base = n == 0 ? 0 : 32;
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = (uint8_t)(quants[n+l+0] + 32);
            const uint8_t q2 = (uint8_t)(quants[n+l+32] + 32);
            const uint8_t q3 = (uint8_t)(quants[n+l+64] + 32);
            const uint8_t q4 = (uint8_t)(quants[n+l+96] + 32);
            ql[ql_base+l] = (uint8_t)((q1 & 15) | ((q3 & 15) << 4));
            ql[ql_base+l+32] = (uint8_t)((q2 & 15) | ((q4 & 15) << 4));
            qh[qh_base+l] = (uint8_t)(((q1 >> 4) & 3) | (((q2 >> 4)&3)<<2) | (((q3>>4)&3)<<4) | (((q4>>4)&3)<<6));
        }
    }
    memcpy(out + 192, scales, 16);
    out[208] = (uint8_t)(f16_scale & 0xffu);
    out[209] = (uint8_t)(f16_scale >> 8);
}

static double independent_q6_dot(const uint8_t *q6, const uint8_t *q8k) {
    const float q6d = qwen_f16_to_f32(qwen_load_u16_le(q6 + 208));
    const float q8d = qwen_load_f32_le(q8k);
    const int8_t *sc = (const int8_t *)(q6 + 192);
    const int8_t *q8 = (const int8_t *)(q8k + 4);
    double sum = 0.0;
    for (int idx = 0; idx < 256; ++idx) {
        const int half = idx >= 128;
        const int r = idx - half * 128;
        const int group = r / 32;
        const int l = r % 32;
        const uint8_t *ql = q6 + half * 64;
        const uint8_t *qh = q6 + 128 + half * 32;
        int q;
        if (group == 0) q = (ql[l] & 15) | (((qh[l] >> 0)&3)<<4);
        else if (group == 1) q = (ql[l+32] & 15) | (((qh[l] >> 2)&3)<<4);
        else if (group == 2) q = (ql[l] >> 4) | (((qh[l] >> 4)&3)<<4);
        else q = (ql[l+32] >> 4) | (((qh[l] >> 6)&3)<<4);
        q -= 32;
        sum += (double)q6d * (double)sc[idx/16] * (double)q * (double)q8d * (double)q8[idx];
    }
    return sum;
}

static double independent_q8_0_dot(const uint8_t *x, const uint8_t *y, size_t n) {
    const size_t nb = n / QWEN_QK8_0;
    double sum = 0.0;
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = x + ib * QWEN_BLOCK_Q8_0;
        const uint8_t *yb = y + ib * QWEN_BLOCK_Q8_0;
        const double dx = qwen_f16_to_f32(qwen_load_u16_le(xb));
        const double dy = qwen_f16_to_f32(qwen_load_u16_le(yb));
        const int8_t *xq = (const int8_t *)(xb + 2);
        const int8_t *yq = (const int8_t *)(yb + 2);
        for (size_t j = 0; j < QWEN_QK8_0; ++j) {
            sum += dx * (double)xq[j] * dy * (double)yq[j];
        }
    }
    return sum;
}

static int check_f16_case(uint16_t bits, float expected) {
    const float got = qwen_f16_to_f32(qwen_load_u16_le((const uint8_t *)&bits));
    if (got != expected) {
        fprintf(stderr, "f16 mismatch bits=0x%04x got=%.9g expected=%.9g\n", bits, got, expected);
        return 0;
    }
    return 1;
}

static int check_f32_to_f16_case(float value, uint16_t expected) {
    const uint16_t got = qwen_f32_to_f16(value);
    if (got != expected) {
        fprintf(stderr, "f32->f16 mismatch value=%.9g got=0x%04x expected=0x%04x\n", value, got, expected);
        return 0;
    }
    return 1;
}

int main(void) {
    if (!qwen_host_little_endian()) {
        fprintf(stderr, "selftest currently requires little-endian host\n");
        return 2;
    }
    if (!check_f16_case(0x0001u, 0x1p-24f) ||
        !check_f16_case(0x0002u, 0x1p-23f) ||
        !check_f16_case(0x03ffu, 1023.0f * 0x1p-24f) ||
        !check_f16_case(0x0400u, 0x1p-14f) ||
        !check_f16_case(0x8001u, -0x1p-24f)) {
        return 9;
    }
    if (!check_f32_to_f16_case(1.0f, 0x3c00u) ||
        !check_f32_to_f16_case(-2.0f, 0xc000u) ||
        !check_f32_to_f16_case(0x1p-14f, 0x0400u) ||
        !check_f32_to_f16_case(0x1p-24f, 0x0001u) ||
        !check_f32_to_f16_case(-0x1p-24f, 0x8001u)) {
        return 10;
    }
    if (qwen_nearest_int(1.5f) != 2 || qwen_nearest_int(2.5f) != 2 ||
        qwen_nearest_int(-1.5f) != -2 || qwen_nearest_int(-2.5f) != -2) {
        return 18;
    }

    /* llama.cpp's optimized x86 Q8_0 activation path uses nearest-even, not
     * the roundf() ties-away behavior of quantize_row_q8_0_ref().  A real
     * Qwen3.8 BOS attn_norm activation contains an exact 22.5 tie, so this is
     * a model-visible correctness requirement rather than a cosmetic detail. */
    float q8_ties[QWEN_QK8_0] = {0};
    q8_ties[0] = 127.0f;
    q8_ties[1] = 22.5f;
    q8_ties[2] = -22.5f;
    q8_ties[3] = 23.5f;
    q8_ties[4] = -23.5f;
    uint8_t q8_tie_block[QWEN_BLOCK_Q8_0];
    if (qwen_quantize_q8_0_scalar(q8_ties, QWEN_QK8_0, q8_tie_block, sizeof(q8_tie_block)) != 0) return 26;
    const int8_t *q8_tie_qs = (const int8_t *)(q8_tie_block + 2);
    if (q8_tie_qs[0] != 127 || q8_tie_qs[1] != 22 || q8_tie_qs[2] != -22 ||
        q8_tie_qs[3] != 24 || q8_tie_qs[4] != -24) {
        fprintf(stderr, "Q8_0 nearest-even tie mismatch: %d %d %d %d %d\n",
                q8_tie_qs[0], q8_tie_qs[1], q8_tie_qs[2], q8_tie_qs[3], q8_tie_qs[4]);
        return 27;
    }

    uint8_t q6[2 * QWEN_BLOCK_Q6_K];
    int8_t scales[16];
    int8_t quants[256];
    for (int i = 0; i < 16; ++i) scales[i] = (int8_t)((i % 2 ? 1 : -1) * (i + 1));
    for (int i = 0; i < 256; ++i) quants[i] = (int8_t)(((i * 17 + 3) % 64) - 32);
    pack_q6_fixture(q6, 0x3400u /* 0.25 */, scales, quants);
    for (int i = 0; i < 16; ++i) scales[i] = (int8_t)(8 - i);
    for (int i = 0; i < 256; ++i) quants[i] = (int8_t)(31 - ((i * 29 + 11) % 64));
    pack_q6_fixture(q6 + QWEN_BLOCK_Q6_K, 0xb000u /* -0.125 */, scales, quants);

    float x[512];
    for (int i = 0; i < 256; ++i) x[i] = (float)((i % 255) - 127) / 127.0f;
    for (int i = 0; i < 256; ++i) x[256+i] = 0.75f * (float)(((i * 7) % 251) - 125) / 125.0f;
    uint8_t q8k[2 * QWEN_BLOCK_Q8_K];
    if (qwen_quantize_q8_k_scalar(x, 512, q8k, sizeof(q8k)) != 0) return 3;

    const double expected = independent_q6_dot(q6, q8k) + independent_q6_dot(q6 + QWEN_BLOCK_Q6_K, q8k + QWEN_BLOCK_Q8_K);
    const float got = qwen_vec_dot_q6_k_q8_k_scalar(q6, sizeof(q6), q8k, sizeof(q8k), 512);
    const double err = fabs((double)got - expected);
    const double limit = 2e-5 * fmax(1.0, fabs(expected));
    if (!(err <= limit)) {
        fprintf(stderr, "Q6_K dot mismatch got=%.9g expected=%.17g err=%.9g limit=%.9g\n", got, expected, err, limit);
        return 4;
    }

    uint8_t q8k_row[QWEN_BLOCK_Q8_K];
    if (qwen_quantize_q8_k_scalar(x, 256, q8k_row, sizeof(q8k_row)) != 0) return 19;
    float q6_mv[2] = {0};
    if (qwen_matvec_q6_k_q8_k_scalar(q6, sizeof(q6), 2, 256, q8k_row, sizeof(q8k_row), q6_mv) != 0) return 20;
    const double q6_mv_e0 = independent_q6_dot(q6, q8k_row);
    const double q6_mv_e1 = independent_q6_dot(q6 + QWEN_BLOCK_Q6_K, q8k_row);
    const double q6_mv_err = fmax(fabs((double)q6_mv[0] - q6_mv_e0), fabs((double)q6_mv[1] - q6_mv_e1));
    if (q6_mv_err > 2e-5 * fmax(1.0, fmax(fabs(q6_mv_e0), fabs(q6_mv_e1)))) return 21;

    float zero[256] = {0};
    uint8_t zq8[QWEN_BLOCK_Q8_K];
    if (qwen_quantize_q8_k_scalar(zero, 256, zq8, sizeof(zq8)) != 0) return 5;
    for (size_t i = 0; i < sizeof(zq8); ++i) if (zq8[i] != 0) return 6;
    const float zdot = qwen_vec_dot_q6_k_q8_k_scalar(q6, QWEN_BLOCK_Q6_K, zq8, sizeof(zq8), 256);
    if (zdot != 0.0f) return 7;
    if (!isnan(qwen_vec_dot_q6_k_q8_k_scalar(q6, QWEN_BLOCK_Q6_K - 1, zq8, sizeof(zq8), 256))) return 8;

    /* Keep amplitudes small enough that d = amax/127 is a genuine FP16
     * subnormal.  This exercises the exact bug class found on real Q6_K
     * super-scales, now on both Q8_0 scale encode and decode. */
    float q8w[64];
    float q8a[64];
    for (int i = 0; i < 64; ++i) {
        q8w[i] = 0.0018f * (float)(((i * 13 + 5) % 63) - 31) / 31.0f;
        q8a[i] = 0.0024f * (float)(((i * 19 + 7) % 61) - 30) / 30.0f;
    }
    uint8_t qw[2 * QWEN_BLOCK_Q8_0];
    uint8_t qa[2 * QWEN_BLOCK_Q8_0];
    if (qwen_quantize_q8_0_scalar(q8w, 64, qw, sizeof(qw)) != 0) return 11;
    if (qwen_quantize_q8_0_scalar(q8a, 64, qa, sizeof(qa)) != 0) return 12;
    int q8_subnormal_scales = 0;
    for (int ib = 0; ib < 2; ++ib) {
        const uint16_t sw = qwen_load_u16_le(qw + ib * QWEN_BLOCK_Q8_0);
        const uint16_t sa = qwen_load_u16_le(qa + ib * QWEN_BLOCK_Q8_0);
        if ((sw & 0x7c00u) == 0 && (sw & 0x03ffu) != 0) ++q8_subnormal_scales;
        if ((sa & 0x7c00u) == 0 && (sa & 0x03ffu) != 0) ++q8_subnormal_scales;
    }
    if (q8_subnormal_scales != 4) return 13;
    const double q8_expected = independent_q8_0_dot(qw, qa, 64);
    const float q8_got = qwen_vec_dot_q8_0_q8_0_scalar(qw, sizeof(qw), qa, sizeof(qa), 64);
    const double q8_err = fabs((double)q8_got - q8_expected);
    const double q8_limit = 2e-6 * fmax(1.0, fabs(q8_expected));
    if (!(q8_err <= q8_limit)) {
        fprintf(stderr, "Q8_0 dot mismatch got=%.9g expected=%.17g err=%.9g limit=%.9g\n", q8_got, q8_expected, q8_err, q8_limit);
        return 14;
    }

    float q8_mv[2] = {0};
    if (qwen_matvec_q8_0_q8_0_scalar(qw, sizeof(qw), 2, 32, qa, QWEN_BLOCK_Q8_0, q8_mv) != 0) return 22;
    const double q8_mv_e0 = independent_q8_0_dot(qw, qa, 32);
    const double q8_mv_e1 = independent_q8_0_dot(qw + QWEN_BLOCK_Q8_0, qa, 32);
    const double q8_mv_err = fmax(fabs((double)q8_mv[0] - q8_mv_e0), fabs((double)q8_mv[1] - q8_mv_e1));
    if (q8_mv_err > 2e-6 * fmax(1.0, fmax(fabs(q8_mv_e0), fabs(q8_mv_e1)))) return 23;

    uint8_t q8zero[QWEN_BLOCK_Q8_0];
    if (qwen_quantize_q8_0_scalar(zero, 32, q8zero, sizeof(q8zero)) != 0) return 15;
    for (size_t i = 0; i < sizeof(q8zero); ++i) if (q8zero[i] != 0) return 16;
    if (!isnan(qwen_vec_dot_q8_0_q8_0_scalar(qw, QWEN_BLOCK_Q8_0 - 1, qa, QWEN_BLOCK_Q8_0, 32))) return 17;
    if (qwen_matvec_q8_0_q8_0_scalar(qw, sizeof(qw) - 1, 2, 32, qa, QWEN_BLOCK_Q8_0, q8_mv) == 0) return 24;
    if (qwen_matvec_q6_k_q8_k_scalar(q6, sizeof(q6) - 1, 2, 256, q8k_row, sizeof(q8k_row), q6_mv) == 0) return 25;

    printf("{\n");
    printf("  \"schema\": \"qwen38-native-quant-sanity-v4\",\n");
    printf("  \"status\": \"PASS\",\n");
    printf("  \"llama_cpp_reference_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n");
    printf("  \"model_weights_downloaded\": false,\n");
    printf("  \"f16_decode_subnormal_cases\": 5,\n");
    printf("  \"f16_encode_cases\": 5,\n");
    printf("  \"q8_k_nearest_int_tie_cases\": 4,\n");
    printf("  \"q8_0_x86_runtime_nearest_even_tie_cases\": 4,\n");
    printf("  \"q6_k_block_bytes\": %d,\n", QWEN_BLOCK_Q6_K);
    printf("  \"q8_k_block_bytes\": %d,\n", QWEN_BLOCK_Q8_K);
    printf("  \"q8_0_block_bytes\": %d,\n", QWEN_BLOCK_Q8_0);
    printf("  \"q6_tested_elements\": 512,\n");
    printf("  \"q6_dot_abs_error\": %.9g,\n", err);
    printf("  \"q6_dot_error_limit\": %.9g,\n", limit);
    printf("  \"q6_matvec_rows\": 2,\n");
    printf("  \"q6_matvec_max_abs_error\": %.9g,\n", q6_mv_err);
    printf("  \"q6_zero_activation_dot\": %.9g,\n", zdot);
    printf("  \"q8_0_tested_elements\": 64,\n");
    printf("  \"q8_0_subnormal_scales\": %d,\n", q8_subnormal_scales);
    printf("  \"q8_0_dot_abs_error\": %.9g,\n", q8_err);
    printf("  \"q8_0_dot_error_limit\": %.9g,\n", q8_limit);
    printf("  \"q8_0_matvec_rows\": 2,\n");
    printf("  \"q8_0_matvec_max_abs_error\": %.9g,\n", q8_mv_err);
    printf("  \"invalid_size_rejected\": true\n");
    printf("}\n");
    return 0;
}
#endif