/* Native Bonsai 2 low-bit kernels for the Qwen3.8 K3 streaming runtime.
 *
 * Semantics are pinned to PrismML llama.cpp prism-b10683-d8f26ee:
 *   PQ2_0  type 142: group 128, fp16 scale + 32 packed 2-bit bytes.
 *   PTQ1_0 type 143: group 128, 24 base-3 bytes + 2 high bytes + fp16 scale.
 * Activations use the existing Q8_0 quantizer. Folded weights require an
 * optional sign vector followed by normalized blockwise Walsh-Hadamard.
 *
 * This file deliberately reuses the proven AVX2 Q8/Q6 bridge helpers so the
 * new path plugs into the same native runtime ABI instead of introducing
 * llama.cpp as a runtime dependency.
 */
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf_quant_dot_avx2.c"

#define QWEN_QK_PQ2_0 128
#define QWEN_BLOCK_PQ2_0 34
#define QWEN_QK_PTQ1_0 128
#define QWEN_BLOCK_PTQ1_0 28

static inline int32_t qwen_bonsai2_dot_i8_32_avx2(
        const int8_t *a, const int8_t *b) {
    const __m128i a0 = _mm_loadu_si128((const __m128i *)(a + 0));
    const __m128i a1 = _mm_loadu_si128((const __m128i *)(a + 16));
    const __m128i b0 = _mm_loadu_si128((const __m128i *)(b + 0));
    const __m128i b1 = _mm_loadu_si128((const __m128i *)(b + 16));
    const __m256i alo = _mm256_cvtepi8_epi16(a0);
    const __m256i ahi = _mm256_cvtepi8_epi16(a1);
    const __m256i blo = _mm256_cvtepi8_epi16(b0);
    const __m256i bhi = _mm256_cvtepi8_epi16(b1);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i p0 = _mm256_madd_epi16(_mm256_mullo_epi16(alo, blo), ones);
    const __m256i p1 = _mm256_madd_epi16(_mm256_mullo_epi16(ahi, bhi), ones);
    return qwen_hsum8_i32(_mm256_add_epi32(p0, p1));
}

static void qwen_bonsai2_ptq1_lut(int8_t lut[256][5]) {
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int byte = 0; byte < 256; ++byte) {
        for (int n = 0; n < 5; ++n) {
            const uint8_t v = (uint8_t)((uint8_t)byte * pow3[n]);
            const int16_t xi = (int16_t)(((uint16_t)v * 3u) >> 8);
            lut[byte][n] = (int8_t)(xi - 1);
        }
    }
}

static void qwen_bonsai2_decode_ptq1_block(
        const uint8_t *block, const int8_t lut[256][5], int8_t q[128]) {
    const uint8_t *qs = block;
    const uint8_t *qh = block + 24;
    int o = 0;

    /* Prism staging for a 24-byte qs payload is 16 then 8 bytes. */
    for (int n = 0; n < 5; ++n) {
        for (int m = 0; m < 16; ++m) q[o++] = lut[qs[m]][n];
    }
    for (int n = 0; n < 5; ++n) {
        for (int m = 16; m < 24; ++m) q[o++] = lut[qs[m]][n];
    }
    for (int n = 0; n < 4; ++n) {
        for (int h = 0; h < 2; ++h) q[o++] = lut[qh[h]][n];
    }
}

static void qwen_bonsai2_decode_pq2_block(
        const uint8_t *block, int8_t q[128]) {
    const uint8_t *qs = block + 2;
    for (int b = 0; b < 32; ++b) {
        const uint8_t v = qs[b];
        q[b*4 + 0] = (int8_t)(((v >> 0) & 3) - 1);
        q[b*4 + 1] = (int8_t)(((v >> 2) & 3) - 1);
        q[b*4 + 2] = (int8_t)(((v >> 4) & 3) - 1);
        q[b*4 + 3] = (int8_t)(((v >> 6) & 3) - 1);
    }
}


int qwen_bonsai2_dequantize_ptq1_0_row(
        const uint8_t *weights, size_t weights_bytes, size_t n, float *out) {
    if (!weights || !out || n == 0 || n % QWEN_QK_PTQ1_0 != 0) return -1;
    const size_t nb = n / QWEN_QK_PTQ1_0;
    if (weights_bytes != nb * QWEN_BLOCK_PTQ1_0) return -2;
    int8_t lut[256][5];
    int8_t q[128];
    qwen_bonsai2_ptq1_lut(lut);
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = weights + ib * QWEN_BLOCK_PTQ1_0;
        qwen_bonsai2_decode_ptq1_block(xb, lut, q);
        const float d = qwen_f16_to_f32(qwen_load_u16_le(xb + 26));
        for (int j = 0; j < 128; ++j) out[ib * 128 + (size_t)j] = d * (float)q[j];
    }
    return 0;
}

int qwen_bonsai2_dequantize_pq2_0_row(
        const uint8_t *weights, size_t weights_bytes, size_t n, float *out) {
    if (!weights || !out || n == 0 || n % QWEN_QK_PQ2_0 != 0) return -1;
    const size_t nb = n / QWEN_QK_PQ2_0;
    if (weights_bytes != nb * QWEN_BLOCK_PQ2_0) return -2;
    int8_t q[128];
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = weights + ib * QWEN_BLOCK_PQ2_0;
        qwen_bonsai2_decode_pq2_block(xb, q);
        const float d = qwen_f16_to_f32(qwen_load_u16_le(xb));
        for (int j = 0; j < 128; ++j) out[ib * 128 + (size_t)j] = d * (float)q[j];
    }
    return 0;
}

static inline float qwen_bonsai2_bf16_to_f32(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t qwen_bonsai2_f32_to_bf16_rne(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((bits >> 16) | 64u);
    }
    return (uint16_t)((bits + (0x7fffu + ((bits >> 16) & 1u))) >> 16);
}

static inline float qwen_bonsai2_vec_dot_bf16_avx2(
        const uint16_t *x, const uint16_t *y, size_t n) {
    size_t i = 0;
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();

#define QWEN_BF16_LOAD8(p) _mm256_castsi256_ps(     _mm256_slli_epi32(         _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))

    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(
            _mm256_mul_ps(QWEN_BF16_LOAD8(x + i), QWEN_BF16_LOAD8(y + i)), c1);
        c2 = _mm256_add_ps(
            _mm256_mul_ps(QWEN_BF16_LOAD8(x + i + 8), QWEN_BF16_LOAD8(y + i + 8)), c2);
        c3 = _mm256_add_ps(
            _mm256_mul_ps(QWEN_BF16_LOAD8(x + i + 16), QWEN_BF16_LOAD8(y + i + 16)), c3);
        c4 = _mm256_add_ps(
            _mm256_mul_ps(QWEN_BF16_LOAD8(x + i + 24), QWEN_BF16_LOAD8(y + i + 24)), c4);
    }

    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3), _mm256_add_ps(c2, c4));
    __m128 g = _mm_add_ps(
        _mm256_extractf128_ps(c1, 1), _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    float sum = _mm_cvtss_f32(g);

#undef QWEN_BF16_LOAD8

    for (; i < n; ++i) {
        sum += qwen_bonsai2_bf16_to_f32(x[i]) * qwen_bonsai2_bf16_to_f32(y[i]);
    }
    return sum;
}

int qwen_bonsai2_matvec_bf16_f32(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const float *activation, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0) return -1;
    if (weights_bytes != rows * n * 2) return -2;

    /* GGML BF16 matmul does not dot BF16 weights directly with F32 src1.
     * Its type trait converts src1 F32 -> BF16 using round-to-nearest-even,
     * then ggml_vec_dot_bf16 consumes BF16 x BF16. Mirror that contract. */
    uint16_t *act_bf16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!act_bf16) return -3;
    for (size_t i = 0; i < n; ++i) {
        act_bf16[i] = qwen_bonsai2_f32_to_bf16_rne(activation[i]);
    }

    for (size_t r = 0; r < rows; ++r) {
        const uint16_t *row = (const uint16_t *)(weights + r * n * 2);
        out[r] = qwen_bonsai2_vec_dot_bf16_avx2(row, act_bf16, n);
    }
    free(act_bf16);
    return 0;
}

/* Qwen3.5 GDN ssm_out receives 48 V heads in tiled order where
 * old_head = k_group + 16 * repeat. Prism folds ssm_out after regrouping to
 * grouped order new_head = repeat + 3 * k_group. Each head has 128 values. */
int qwen_bonsai2_permute_gdn_ssm_out_f32(
        const float *src, float *dst, size_t n,
        size_t head_dim, size_t n_k, size_t repeat) {
    if (!src || !dst || head_dim == 0 || n_k == 0 || repeat == 0) return -1;
    if (n != head_dim * n_k * repeat) return -2;
    for (size_t k = 0; k < n_k; ++k) {
        for (size_t rep = 0; rep < repeat; ++rep) {
            const size_t old_head = k + n_k * rep;
            const size_t new_head = rep + repeat * k;
            memcpy(
                dst + new_head * head_dim,
                src + old_head * head_dim,
                head_dim * sizeof(float));
        }
    }
    return 0;
}

static float qwen_bonsai2_vec_dot_pq2_q8_0(
        const uint8_t *weights, const uint8_t *activation, size_t n) {
    const size_t nb = n / QWEN_QK_PQ2_0;
    float sumf = 0.0f;
    int8_t q[128];

    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = weights + ib * QWEN_BLOCK_PQ2_0;
        const uint8_t *yb = activation + ib * 4 * QWEN_BLOCK_Q8_0;
        qwen_bonsai2_decode_pq2_block(xb, q);
        const float d0 = qwen_f16_to_f32(qwen_load_u16_le(xb));

        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const uint8_t *ab = yb + k * QWEN_BLOCK_Q8_0;
            const float d1 = qwen_f16_to_f32(qwen_load_u16_le(ab));
            const int32_t dot = qwen_bonsai2_dot_i8_32_avx2(
                q + k * 32, (const int8_t *)(ab + 2));
            sumi += d1 * (float)dot;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

static float qwen_bonsai2_vec_dot_ptq1_q8_0(
        const uint8_t *weights, const uint8_t *activation, size_t n,
        const int8_t lut[256][5]) {
    const size_t nb = n / QWEN_QK_PTQ1_0;
    float sumf = 0.0f;
    int8_t q[128];

    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = weights + ib * QWEN_BLOCK_PTQ1_0;
        const uint8_t *yb = activation + ib * 4 * QWEN_BLOCK_Q8_0;
        qwen_bonsai2_decode_ptq1_block(xb, lut, q);
        const float d0 = qwen_f16_to_f32(qwen_load_u16_le(xb + 26));

        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const uint8_t *ab = yb + k * QWEN_BLOCK_Q8_0;
            const float d1 = qwen_f16_to_f32(qwen_load_u16_le(ab));
            const int32_t dot = qwen_bonsai2_dot_i8_32_avx2(
                q + k * 32, (const int8_t *)(ab + 2));
            sumi += d1 * (float)dot;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

int qwen_bonsai2_matvec_pq2_0_q8_0(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 ||
        n % QWEN_QK_PQ2_0 != 0) return -1;
    const size_t wr = (n / QWEN_QK_PQ2_0) * QWEN_BLOCK_PQ2_0;
    const size_t ar = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar) return -2;
    if (weights_bytes != rows * wr) return -3;

    for (size_t r = 0; r < rows; ++r) {
        out[r] = qwen_bonsai2_vec_dot_pq2_q8_0(
            weights + r * wr, activation, n);
    }
    return 0;
}

int qwen_bonsai2_matvec_ptq1_0_q8_0(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 ||
        n % QWEN_QK_PTQ1_0 != 0) return -1;
    const size_t wr = (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ar = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar) return -2;
    if (weights_bytes != rows * wr) return -3;

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    for (size_t r = 0; r < rows; ++r) {
        out[r] = qwen_bonsai2_vec_dot_ptq1_q8_0(
            weights + r * wr, activation, n, lut);
    }
    return 0;
}

/* Prism applies optional element signs, then a normalized Sylvester-Walsh
 * Hadamard independently to every block_size-wide slice of the activation.
 * The transform is self-inverse when signs are identity. */
int qwen_bonsai2_fwht_blocks(
        float *x, size_t n, size_t block_size, const int8_t *signs) {
    if (!x || n == 0 || block_size == 0 ||
        (block_size & (block_size - 1)) != 0 || n % block_size != 0) {
        return -1;
    }
    const float scale = 1.0f / sqrtf((float)block_size);
    for (size_t base = 0; base < n; base += block_size) {
        for (size_t j = 0; j < block_size; ++j) {
            float v = x[base + j];
            if (signs) v *= (float)signs[base + j];
            x[base + j] = v * scale;
        }
        for (size_t len = 1; len < block_size; len <<= 1) {
            for (size_t i = 0; i < block_size; i += 2 * len) {
                for (size_t j = 0; j < len; ++j) {
                    const float u = x[base + i + j];
                    const float v = x[base + i + len + j];
                    x[base + i + j] = u + v;
                    x[base + i + len + j] = u - v;
                }
            }
        }
    }
    return 0;
}


int qwen_bonsai2_inverse_fwht_blocks(
        float *x, size_t n, size_t block_size, const int8_t *signs) {
    if (!x || n == 0 || block_size == 0 ||
        (block_size & (block_size - 1)) != 0 || n % block_size != 0) {
        return -1;
    }
    const float scale = 1.0f / sqrtf((float)block_size);
    for (size_t base = 0; base < n; base += block_size) {
        for (size_t j = 0; j < block_size; ++j) x[base + j] *= scale;
        for (size_t len = 1; len < block_size; len <<= 1) {
            for (size_t i = 0; i < block_size; i += 2 * len) {
                for (size_t j = 0; j < len; ++j) {
                    const float u = x[base + i + j];
                    const float v = x[base + i + len + j];
                    x[base + i + j] = u + v;
                    x[base + i + len + j] = u - v;
                }
            }
        }
        if (signs) {
            for (size_t j = 0; j < block_size; ++j) {
                x[base + j] *= (float)signs[base + j];
            }
        }
    }
    return 0;
}

#ifdef QWEN_BONSAI2_QUANT_SELFTEST
static int qwen_bonsai2_close(float a, float b, float tol) {
    return fabsf(a - b) <= tol * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
}

int main(void) {
    /* Both low-bit encodings decode an all-zero payload to ternary -1. */
    uint8_t pq[QWEN_BLOCK_PQ2_0] = {0};
    uint8_t tq[QWEN_BLOCK_PTQ1_0] = {0};
    pq[0] = 0x00; pq[1] = 0x3c;       /* fp16 1.0 */
    tq[26] = 0x00; tq[27] = 0x3c;     /* fp16 1.0 */

    uint8_t act[4 * QWEN_BLOCK_Q8_0] = {0};
    for (int k = 0; k < 4; ++k) {
        uint8_t *b = act + k * QWEN_BLOCK_Q8_0;
        b[0] = 0x00; b[1] = 0x3c;     /* fp16 1.0 */
        memset(b + 2, 1, 32);
    }

    float out_pq = 0.0f, out_tq = 0.0f;
    if (qwen_bonsai2_matvec_pq2_0_q8_0(
            pq, sizeof(pq), 1, 128, act, sizeof(act), &out_pq) != 0) return 2;
    if (qwen_bonsai2_matvec_ptq1_0_q8_0(
            tq, sizeof(tq), 1, 128, act, sizeof(act), &out_tq) != 0) return 3;
    if (out_pq != -128.0f || out_tq != -128.0f) {
        fprintf(stderr, "ternary dot mismatch pq=%.9g ptq=%.9g\n", out_pq, out_tq);
        return 4;
    }

    float h[4] = {1, 2, 3, 4};
    if (qwen_bonsai2_fwht_blocks(h, 4, 4, NULL) != 0) return 5;
    const float expected[4] = {5.0f, -1.0f, -2.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        if (!qwen_bonsai2_close(h[i], expected[i], 1e-6f)) {
            fprintf(stderr, "FWHT mismatch i=%d got=%.9g expected=%.9g\n",
                    i, h[i], expected[i]);
            return 6;
        }
    }

    int8_t signs[4] = {1, -1, 1, -1};
    float hs[4] = {1, 2, 3, 4};
    if (qwen_bonsai2_fwht_blocks(hs, 4, 4, signs) != 0) return 7;
    const float expected_s[4] = {-1.0f, 5.0f, 0.0f, -2.0f};
    for (int i = 0; i < 4; ++i) {
        if (!qwen_bonsai2_close(hs[i], expected_s[i], 1e-6f)) {
            fprintf(stderr, "signed FWHT mismatch i=%d got=%.9g expected=%.9g\n",
                    i, hs[i], expected_s[i]);
            return 8;
        }
    }

    float inv[4] = {5.0f, -1.0f, -2.0f, 0.0f};
    if (qwen_bonsai2_inverse_fwht_blocks(inv, 4, 4, NULL) != 0) return 9;
    const float primal[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) {
        if (!qwen_bonsai2_close(inv[i], primal[i], 1e-6f)) return 10;
    }

    float tiled[12], grouped[12];
    for (int i = 0; i < 12; ++i) tiled[i] = (float)i;
    if (qwen_bonsai2_permute_gdn_ssm_out_f32(tiled, grouped, 12, 2, 2, 3) != 0) return 11;
    const float grouped_expected[12] = {0,1,4,5,8,9,2,3,6,7,10,11};
    for (int i = 0; i < 12; ++i) {
        if (grouped[i] != grouped_expected[i]) return 12;
    }

    puts("QWEN38_BONSAI2_NATIVE_C_SELFTEST_PASS");
    return 0;
}
#endif
