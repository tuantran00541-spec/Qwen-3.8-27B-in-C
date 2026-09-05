/* Exact Q4_0 x Q8_0 bridge for the Qwen3.8 MTP block.
 *
 * The proven trunk bridge deliberately supports only Q6_K/Q8_0.  Qwen3.8's
 * auxiliary MTP block uses Q4_0 for its eight large matrices, so keep that new
 * path isolated here until a real MTP gate is green.
 *
 * The scalar reference mirrors GGML's generic q4_0/q8_0 dot-product order:
 * one integer dot per 32-value block, then one FP32 scaled contribution added
 * to the running sum.  The AVX2 path only vectorizes the integer dot and keeps
 * the same per-block FP32 accumulation order.  Compile with -fno-fast-math and
 * -ffp-contract=off for bitwise A/B testing.
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gguf_quant_dot.c"

#define QWEN_BLOCK_Q4_0 18

float qwen_vec_dot_q4_0_q8_0_reference(
        const uint8_t *q4, size_t q4_bytes,
        const uint8_t *q8, size_t q8_bytes, size_t n) {
    if (!q4 || !q8 || n == 0 || n % QWEN_QK8_0 != 0) return NAN;
    const size_t nb = n / QWEN_QK8_0;
    if (q4_bytes != nb * QWEN_BLOCK_Q4_0 || q8_bytes != nb * QWEN_BLOCK_Q8_0) return NAN;

    float sum = 0.0f;
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = q4 + ib * QWEN_BLOCK_Q4_0;
        const uint8_t *yb = q8 + ib * QWEN_BLOCK_Q8_0;
        const uint8_t *xq = xb + 2;
        const int8_t *yq = (const int8_t *)(yb + 2);
        int32_t sumi0 = 0;
        int32_t sumi1 = 0;
        for (int j = 0; j < 16; ++j) {
            const int32_t v0 = (int32_t)(xq[j] & 0x0fu) - 8;
            const int32_t v1 = (int32_t)(xq[j] >> 4) - 8;
            sumi0 += v0 * (int32_t)yq[j];
            sumi1 += v1 * (int32_t)yq[j + 16];
        }
        const int32_t sumi = sumi0 + sumi1;
        const float dx = qwen_f16_to_f32(qwen_load_u16_le(xb));
        const float dy = qwen_f16_to_f32(qwen_load_u16_le(yb));
        sum += (float)sumi * dx * dy;
    }
    return sum;
}

static inline int32_t qwen_q4_hsum8_i32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

float qwen_vec_dot_q4_0_q8_0_scalar(
        const uint8_t *q4, size_t q4_bytes,
        const uint8_t *q8, size_t q8_bytes, size_t n) {
    if (!q4 || !q8 || n == 0 || n % QWEN_QK8_0 != 0) return NAN;
    const size_t nb = n / QWEN_QK8_0;
    if (q4_bytes != nb * QWEN_BLOCK_Q4_0 || q8_bytes != nb * QWEN_BLOCK_Q8_0) return NAN;

    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i eight = _mm_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    float sum = 0.0f;

    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = q4 + ib * QWEN_BLOCK_Q4_0;
        const uint8_t *yb = q8 + ib * QWEN_BLOCK_Q8_0;

        const __m128i packed = _mm_loadu_si128((const __m128i *)(xb + 2));
        const __m128i lo4 = _mm_sub_epi8(_mm_and_si128(packed, mask), eight);
        const __m128i hi4 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed, 4), mask), eight);
        const __m128i y0 = _mm_loadu_si128((const __m128i *)(yb + 2));
        const __m128i y1 = _mm_loadu_si128((const __m128i *)(yb + 18));

        const __m256i xlo = _mm256_cvtepi8_epi16(lo4);
        const __m256i xhi = _mm256_cvtepi8_epi16(hi4);
        const __m256i ylo = _mm256_cvtepi8_epi16(y0);
        const __m256i yhi = _mm256_cvtepi8_epi16(y1);

        const __m256i p0 = _mm256_madd_epi16(_mm256_mullo_epi16(xlo, ylo), ones);
        const __m256i p1 = _mm256_madd_epi16(_mm256_mullo_epi16(xhi, yhi), ones);
        const int32_t sumi = qwen_q4_hsum8_i32(_mm256_add_epi32(p0, p1));

        const float dx = qwen_f16_to_f32(qwen_load_u16_le(xb));
        const float dy = qwen_f16_to_f32(qwen_load_u16_le(yb));
        sum += (float)sumi * dx * dy;
    }
    return sum;
}

int qwen_matvec_q4_0_q8_0_reference(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t wr = (n / QWEN_QK8_0) * QWEN_BLOCK_Q4_0;
    const size_t ar = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar) return -2;
    if (weights_bytes % wr != 0 || weights_bytes / wr != rows) return -3;
    for (size_t row = 0; row < rows; ++row) {
        out[row] = qwen_vec_dot_q4_0_q8_0_reference(
            weights + row * wr, wr, activation, activation_bytes, n);
    }
    return 0;
}

int qwen_matvec_q4_0_q8_0_scalar(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t wr = (n / QWEN_QK8_0) * QWEN_BLOCK_Q4_0;
    const size_t ar = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar) return -2;
    if (weights_bytes % wr != 0 || weights_bytes / wr != rows) return -3;
    for (size_t row = 0; row < rows; ++row) {
        out[row] = qwen_vec_dot_q4_0_q8_0_scalar(
            weights + row * wr, wr, activation, activation_bytes, n);
    }
    return 0;
}

#ifdef QWEN_Q4_AVX2_EXACT_SELFTEST
static void qwen_pack_q4_fixture(uint8_t out[QWEN_BLOCK_Q4_0], uint16_t scale, const int8_t q[32]) {
    out[0] = (uint8_t)(scale & 0xffu);
    out[1] = (uint8_t)(scale >> 8);
    for (int j = 0; j < 16; ++j) {
        const uint8_t lo = (uint8_t)(q[j] + 8) & 0x0fu;
        const uint8_t hi = (uint8_t)(q[j + 16] + 8) & 0x0fu;
        out[2 + j] = (uint8_t)(lo | (hi << 4));
    }
}

int main(void) {
    uint32_t s = 0x9e3779b9u;

    for (int trial = 0; trial < 400; ++trial) {
        enum { N = 256, NB = N / 32 };
        uint8_t q4[NB * QWEN_BLOCK_Q4_0];
        float x[N];
        uint8_t q8[NB * QWEN_BLOCK_Q8_0];

        for (int ib = 0; ib < NB; ++ib) {
            int8_t q[32];
            for (int j = 0; j < 32; ++j) {
                s = s * 1664525u + 1013904223u;
                q[j] = (int8_t)((s >> 24) % 16 - 8);
            }
            s = s * 1664525u + 1013904223u;
            const float scale = ((int32_t)(s >> 8) % 2001 - 1000) / 997.0f;
            qwen_pack_q4_fixture(q4 + ib * QWEN_BLOCK_Q4_0, qwen_f32_to_f16(scale), q);
        }
        for (int j = 0; j < N; ++j) {
            s = s * 1664525u + 1013904223u;
            x[j] = ((int32_t)(s >> 8) % 40001 - 20000) / 911.0f;
        }
        if (qwen_quantize_q8_0_scalar(x, N, q8, sizeof(q8)) != 0) return 2;

        const float a = qwen_vec_dot_q4_0_q8_0_reference(q4, sizeof(q4), q8, sizeof(q8), N);
        const float b = qwen_vec_dot_q4_0_q8_0_scalar(q4, sizeof(q4), q8, sizeof(q8), N);
        if (memcmp(&a, &b, sizeof(float)) != 0) {
            fprintf(stderr, "Q4 dot mismatch trial=%d ref=%.9g avx2=%.9g\n", trial, a, b);
            return 3;
        }
    }

    {
        enum { N = 256, ROWS = 3, NB = N / 32 };
        uint8_t weights[ROWS * NB * QWEN_BLOCK_Q4_0];
        float x[N];
        uint8_t q8[NB * QWEN_BLOCK_Q8_0];
        for (int row = 0; row < ROWS; ++row) {
            for (int ib = 0; ib < NB; ++ib) {
                int8_t q[32];
                for (int j = 0; j < 32; ++j) {
                    s = s * 1664525u + 1013904223u;
                    q[j] = (int8_t)((s >> 25) % 16 - 8);
                }
                const float scale = (row + 1) * (ib % 2 ? -0.0625f : 0.03125f);
                qwen_pack_q4_fixture(
                    weights + (row * NB + ib) * QWEN_BLOCK_Q4_0,
                    qwen_f32_to_f16(scale), q);
            }
        }
        for (int j = 0; j < N; ++j) x[j] = (float)((j * 37) % 257 - 128) / 73.0f;
        if (qwen_quantize_q8_0_scalar(x, N, q8, sizeof(q8)) != 0) return 4;
        float ref[ROWS] = {0};
        float avx[ROWS] = {0};
        if (qwen_matvec_q4_0_q8_0_reference(weights, sizeof(weights), ROWS, N, q8, sizeof(q8), ref) != 0) return 5;
        if (qwen_matvec_q4_0_q8_0_scalar(weights, sizeof(weights), ROWS, N, q8, sizeof(q8), avx) != 0) return 6;
        if (memcmp(ref, avx, sizeof(ref)) != 0) {
            fprintf(stderr, "Q4 matvec mismatch\n");
            return 7;
        }
    }

    puts("QWEN38_Q4_AVX2_SINGLE_THREAD_BITWISE_PASS");
    return 0;
}
#endif
