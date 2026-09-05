/* Exact multi-vector overlay for Qwen3.8 prefill research.
 *
 * This keeps the proven per-vector floating-point accumulation order but
 * inverts the matrix traversal so one quantized weight row/block serves
 * several already-quantized activations before moving on.  Q6_K unpacking is
 * hoisted across vectors.  The existing one-vector ABI remains untouched.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "gguf_quant_dot_avx2.c"

static int qwen_matvec_many_q8_0_q8_0_exact(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    if (!weights || !activations || !out || rows == 0 || n_vec == 0 || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t nb = n / QWEN_QK8_0;
    const size_t row_bytes = nb * QWEN_BLOCK_Q8_0;
    if (activation_bytes_each != row_bytes) return -2;
    if (weights_bytes != rows * row_bytes) return -3;

    for (size_t r = 0; r < rows; ++r) {
        float *sums = (float *) calloc(n_vec, sizeof(float));
        if (!sums) return -4;
        const uint8_t *wrow = weights + r * row_bytes;
        for (size_t ib = 0; ib < nb; ++ib) {
            const uint8_t *xb = wrow + ib * QWEN_BLOCK_Q8_0;
            const __m128i x0 = _mm_loadu_si128((const __m128i *)(xb + 2));
            const __m128i x1 = _mm_loadu_si128((const __m128i *)(xb + 18));
            const __m256i xlo = _mm256_cvtepi8_epi16(x0);
            const __m256i xhi = _mm256_cvtepi8_epi16(x1);
            const float dx = qwen_f16_to_f32(qwen_load_u16_le(xb));
            const __m256i ones = _mm256_set1_epi16(1);
            for (size_t v = 0; v < n_vec; ++v) {
                const uint8_t *yb = activations + v * activation_bytes_each + ib * QWEN_BLOCK_Q8_0;
                const __m128i y0 = _mm_loadu_si128((const __m128i *)(yb + 2));
                const __m128i y1 = _mm_loadu_si128((const __m128i *)(yb + 18));
                const __m256i ylo = _mm256_cvtepi8_epi16(y0);
                const __m256i yhi = _mm256_cvtepi8_epi16(y1);
                __m256i p0 = _mm256_madd_epi16(_mm256_mullo_epi16(xlo, ylo), ones);
                __m256i p1 = _mm256_madd_epi16(_mm256_mullo_epi16(xhi, yhi), ones);
                const int32_t sumi = qwen_hsum8_i32(_mm256_add_epi32(p0, p1));
                const float dy = qwen_f16_to_f32(qwen_load_u16_le(yb));
                sums[v] += (float)sumi * dx * dy;
            }
        }
        for (size_t v = 0; v < n_vec; ++v) out[v * rows + r] = sums[v];
        free(sums);
    }
    return 0;
}

static int qwen_matvec_many_q6_k_q8_k_exact(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    if (!weights || !activations || !out || rows == 0 || n_vec == 0 || n == 0 || n % QWEN_QK_K != 0) return -1;
    const size_t nb = n / QWEN_QK_K;
    const size_t wr = nb * QWEN_BLOCK_Q6_K;
    const size_t ar = nb * QWEN_BLOCK_Q8_K;
    if (activation_bytes_each != ar) return -2;
    if (weights_bytes != rows * wr) return -3;

    float *sums = (float *) calloc(n_vec * 8, sizeof(float));
    if (!sums) return -4;
    int8_t aux8[QWEN_QK_K];

    for (size_t r = 0; r < rows; ++r) {
        memset(sums, 0, n_vec * 8 * sizeof(float));
        const uint8_t *wrow = weights + r * wr;
        for (size_t ib = 0; ib < nb; ++ib) {
            const uint8_t *xb = wrow + ib * QWEN_BLOCK_Q6_K;
            const uint8_t *ql = xb;
            const uint8_t *qh = xb + 128;
            const int8_t *scales = (const int8_t *)(xb + 192);
            int8_t *a = aux8;
            for (int j = 0; j < QWEN_QK_K; j += 128) {
                for (int l = 0; l < 32; ++l) {
                    a[l+0]  = (int8_t)(((ql[l+0]&15)  | (((qh[l]>>0)&3)<<4))-32);
                    a[l+32] = (int8_t)(((ql[l+32]&15) | (((qh[l]>>2)&3)<<4))-32);
                    a[l+64] = (int8_t)(((ql[l+0]>>4)  | (((qh[l]>>4)&3)<<4))-32);
                    a[l+96] = (int8_t)(((ql[l+32]>>4) | (((qh[l]>>6)&3)<<4))-32);
                }
                a += 128; ql += 64; qh += 32;
            }
            const float dx = qwen_f16_to_f32(qwen_load_u16_le(xb + 208));

            for (size_t v = 0; v < n_vec; ++v) {
                const uint8_t *yb = activations + v * activation_bytes_each + ib * QWEN_BLOCK_Q8_K;
                const int8_t *q8 = (const int8_t *)(yb + 4);
                __m256i lanes = _mm256_setzero_si256();
                a = aux8;
                for (int j = 0; j < QWEN_QK_K/16; ++j) {
                    const __m256i scale = _mm256_set1_epi32((int)scales[j]);
                    __m256i av = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)a));
                    __m256i qv = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)q8));
                    lanes = _mm256_add_epi32(lanes, _mm256_mullo_epi32(scale, _mm256_mullo_epi32(av, qv)));
                    a += 8; q8 += 8;
                    av = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)a));
                    qv = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)q8));
                    lanes = _mm256_add_epi32(lanes, _mm256_mullo_epi32(scale, _mm256_mullo_epi32(av, qv)));
                    a += 8; q8 += 8;
                }
                int32_t lanev[8];
                _mm256_storeu_si256((__m256i *)lanev, lanes);
                const float d = dx * qwen_load_f32_le(yb);
                for (int l = 0; l < 8; ++l) sums[v * 8 + l] += d * (float)lanev[l];
            }
        }
        for (size_t v = 0; v < n_vec; ++v) {
            float sumf = 0.0f;
            for (int l = 0; l < 8; ++l) sumf += sums[v * 8 + l];
            out[v * rows + r] = sumf;
        }
    }
    free(sums);
    return 0;
}

#ifdef QWEN_MATVEC_MANY_SELFTEST
static uint32_t rng_state = 0x8e91b37du;
static uint32_t rng_u32(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f32(void) { return ((int32_t)(rng_u32() >> 8) % 20001 - 10000) / 997.0f; }

static void pack_q6_fixture(uint8_t out[QWEN_BLOCK_Q6_K]) {
    int8_t scales[16], quants[256];
    for (int i = 0; i < 16; ++i) scales[i] = (int8_t)((rng_u32() >> 24) % 31 - 15);
    for (int i = 0; i < 256; ++i) quants[i] = (int8_t)((rng_u32() >> 24) % 64 - 32);
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
    const uint16_t d = qwen_f32_to_f16(0.03125f);
    out[208] = (uint8_t)(d & 0xffu); out[209] = (uint8_t)(d >> 8);
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { NV = 8, NQ6 = 5120, NQ8 = 5120, ROWS = 512 };
    const size_t q6_wr = (NQ6 / QWEN_QK_K) * QWEN_BLOCK_Q6_K;
    const size_t q6_ar = (NQ6 / QWEN_QK_K) * QWEN_BLOCK_Q8_K;
    const size_t q8_wr = (NQ8 / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;

    uint8_t *w6 = (uint8_t *) malloc(ROWS * q6_wr);
    uint8_t *a6 = (uint8_t *) malloc(NV * q6_ar);
    uint8_t *w8 = (uint8_t *) malloc(ROWS * q8_wr);
    uint8_t *a8 = (uint8_t *) malloc(NV * q8_wr);
    float *ref = (float *) malloc(NV * ROWS * sizeof(float));
    float *many = (float *) malloc(NV * ROWS * sizeof(float));
    float *x = (float *) malloc(NQ6 * sizeof(float));
    if (!w6 || !a6 || !w8 || !a8 || !ref || !many || !x) return 2;

    for (size_t r = 0; r < ROWS; ++r)
        for (size_t ib = 0; ib < NQ6 / QWEN_QK_K; ++ib)
            pack_q6_fixture(w6 + r * q6_wr + ib * QWEN_BLOCK_Q6_K);
    for (int v = 0; v < NV; ++v) {
        for (int i = 0; i < NQ6; ++i) x[i] = rng_f32();
        if (qwen_quantize_q8_k_scalar(x, NQ6, a6 + v*q6_ar, q6_ar) != 0) return 3;
    }

    double t0 = now_s();
    for (int v = 0; v < NV; ++v)
        if (qwen_matvec_q6_k_q8_k_scalar(w6, ROWS*q6_wr, ROWS, NQ6, a6+v*q6_ar, q6_ar, ref+v*ROWS) != 0) return 4;
    double seq6 = now_s() - t0;
    t0 = now_s();
    if (qwen_matvec_many_q6_k_q8_k_exact(w6, ROWS*q6_wr, ROWS, NQ6, a6, q6_ar, NV, many) != 0) return 5;
    double many6 = now_s() - t0;
    if (memcmp(ref, many, NV*ROWS*sizeof(float)) != 0) { fprintf(stderr, "Q6 many mismatch\n"); return 6; }

    for (size_t r = 0; r < ROWS; ++r) {
        for (int i = 0; i < NQ8; ++i) x[i] = rng_f32();
        if (qwen_quantize_q8_0_scalar(x, NQ8, w8+r*q8_wr, q8_wr) != 0) return 7;
    }
    for (int v = 0; v < NV; ++v) {
        for (int i = 0; i < NQ8; ++i) x[i] = rng_f32();
        if (qwen_quantize_q8_0_scalar(x, NQ8, a8+v*q8_wr, q8_wr) != 0) return 8;
    }
    t0 = now_s();
    for (int v = 0; v < NV; ++v)
        if (qwen_matvec_q8_0_q8_0_scalar(w8, ROWS*q8_wr, ROWS, NQ8, a8+v*q8_wr, q8_wr, ref+v*ROWS) != 0) return 9;
    double seq8 = now_s() - t0;
    t0 = now_s();
    if (qwen_matvec_many_q8_0_q8_0_exact(w8, ROWS*q8_wr, ROWS, NQ8, a8, q8_wr, NV, many) != 0) return 10;
    double many8 = now_s() - t0;
    if (memcmp(ref, many, NV*ROWS*sizeof(float)) != 0) { fprintf(stderr, "Q8 many mismatch\n"); return 11; }

    printf("Q6 sequential=%.6f many=%.6f speedup=%.4fx\n", seq6, many6, seq6/many6);
    printf("Q8 sequential=%.6f many=%.6f speedup=%.4fx\n", seq8, many8, seq8/many8);
    puts("QWEN38_MATVEC_MANY_BITWISE_PASS");
    free(w6); free(a6); free(w8); free(a8); free(ref); free(many); free(x);
    return 0;
}
#endif
