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
#include "attention_core_exact.c"
#include "gdn_repeat_scale_exact.c"

#define QWEN_QK_PQ2_0 128
#define QWEN_BLOCK_PQ2_0 34
#define QWEN_QK_PTQ1_0 128
#define QWEN_BLOCK_PTQ1_0 28

#ifndef QWEN_EXPORT
#ifdef _WIN32
#define QWEN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define QWEN_EXPORT __attribute__((visibility("default")))
#else
#define QWEN_EXPORT
#endif
#endif

QWEN_EXPORT int qwen_bonsai2_attention_core_f32(
        const float *q,
        size_t q_heads,
        size_t kv_heads,
        size_t head_dim,
        const float *k_cache,
        const float *v_cache,
        size_t n_ctx,
        double scale,
        float *out) {
    return qwen_attention_core_f32_exact(
        q, q_heads, kv_heads, head_dim,
        k_cache, v_cache, n_ctx, scale, out);
}

QWEN_EXPORT int qwen_bonsai2_gdn_repeat_scale_f32(
        const float *q,
        const float *k,
        size_t key_dim,
        size_t repeats,
        float scale,
        float *q_out,
        float *k_out) {
    return qwen38_gdn_repeat_scale_many_exact_f32(
        q, k, 1, key_dim, repeats, scale, q_out, k_out);
}

QWEN_EXPORT int qwen_bonsai2_swiglu_f32(
        const float *gate, const float *up, size_t n, float *out) {
    if (!gate || !up || !out || n == 0) return -1;
    for (size_t i = 0; i < n; ++i) {
        const float x = gate[i];
        const float e = expf(-x);
        const float denom = 1.0f + e;
        const float sigmoid = 1.0f / denom;
        const float silu = x * sigmoid;
        out[i] = silu * up[i];
    }
    return 0;
}


static inline float qwen_bonsai2_round_add_f32(float a, float b) {
    volatile float r = a + b;
    return r;
}

static inline float qwen_bonsai2_round_mul_f32(float a, float b) {
    volatile float r = a * b;
    return r;
}

static inline float qwen_bonsai2_round_div_f32(float a, float b) {
    volatile float r = a / b;
    return r;
}

static inline float qwen_bonsai2_sigmoid_f32_exact(float x) {
    const float e = expf(-x);
    const float denom = qwen_bonsai2_round_add_f32(1.0f, e);
    return qwen_bonsai2_round_div_f32(1.0f, denom);
}

static inline float qwen_bonsai2_silu_f32_exact(float x) {
    return qwen_bonsai2_round_mul_f32(
        x, qwen_bonsai2_sigmoid_f32_exact(x));
}

QWEN_EXPORT int qwen_bonsai2_attention_gate_f32(
        const float *pregate,
        const float *gate,
        size_t n,
        float *out) {
    if (!pregate || !gate || !out || n == 0) return -1;
    for (size_t i = 0; i < n; ++i) {
        out[i] = qwen_bonsai2_round_mul_f32(
            pregate[i], qwen_bonsai2_sigmoid_f32_exact(gate[i]));
    }
    return 0;
}

QWEN_EXPORT int qwen_bonsai2_residual_add_f32(
        const float *a,
        const float *b,
        size_t n,
        float *out) {
    if (!a || !b || !out || n == 0) return -1;
    for (size_t i = 0; i < n; ++i) {
        out[i] = qwen_bonsai2_round_add_f32(a[i], b[i]);
    }
    return 0;
}

QWEN_EXPORT int qwen_bonsai2_gdn_conv_silu_f32(
        const float *qkv,
        const float *history,
        size_t history_count,
        const float *kernels,
        size_t n,
        float *out) {
    if (!qkv || !kernels || !out || n == 0 || history_count > 3) return -1;
    if (history_count > 0 && !history) return -2;

    for (size_t cidx = 0; cidx < n; ++cidx) {
        float cur = qwen_bonsai2_round_mul_f32(
            qkv[cidx], kernels[cidx * 4 + 3]);
        for (size_t lag = 1; lag <= history_count; ++lag) {
            const size_t hist_index = history_count - lag;
            const float prior = history[hist_index * n + cidx];
            const float term = qwen_bonsai2_round_mul_f32(
                prior, kernels[cidx * 4 + (3 - lag)]);
            cur = qwen_bonsai2_round_add_f32(cur, term);
        }
        out[cidx] = qwen_bonsai2_silu_f32_exact(cur);
    }
    return 0;
}

QWEN_EXPORT int qwen_bonsai2_rms_norm_f32(
        const float *x,
        const float *weight,
        size_t rows,
        size_t width,
        float eps,
        float *out) {
    if (!x || !weight || !out || rows == 0 || width == 0) return -1;

    for (size_t row = 0; row < rows; ++row) {
        const size_t base = row * width;
        double sum_sq = 0.0;
        for (size_t i = 0; i < width; ++i) {
            const float v = x[base + i];
            const float sq = qwen_bonsai2_round_mul_f32(v, v);
            sum_sq += (double)sq;
        }
        const float mean = (float)(sum_sq / (double)width);
        const float mean_eps = qwen_bonsai2_round_add_f32(mean, eps);
        const float root = sqrtf(mean_eps);
        const float scale = qwen_bonsai2_round_div_f32(1.0f, root);

        for (size_t i = 0; i < width; ++i) {
            const float scaled = qwen_bonsai2_round_mul_f32(
                x[base + i], scale);
            out[base + i] = qwen_bonsai2_round_mul_f32(
                scaled, weight[i]);
        }
    }
    return 0;
}

QWEN_EXPORT int qwen_bonsai2_gdn_norm_gate_f32(
        const float *core,
        const float *norm_weight,
        const float *z,
        size_t heads,
        size_t head_dim,
        float eps,
        float *out) {
    if (!core || !norm_weight || !z || !out ||
        heads == 0 || head_dim == 0) {
        return -1;
    }

    for (size_t h = 0; h < heads; ++h) {
        const size_t base = h * head_dim;
        double sum_sq = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            const float v = core[base + d];
            const float sq = qwen_bonsai2_round_mul_f32(v, v);
            sum_sq += (double)sq;
        }
        const float mean = (float)(sum_sq / (double)head_dim);
        const float mean_eps = qwen_bonsai2_round_add_f32(mean, eps);
        const float root = sqrtf(mean_eps);
        const float scale = qwen_bonsai2_round_div_f32(1.0f, root);

        for (size_t d = 0; d < head_dim; ++d) {
            const size_t idx = base + d;
            const float scaled = qwen_bonsai2_round_mul_f32(
                core[idx], scale);
            const float normalized = qwen_bonsai2_round_mul_f32(
                scaled, norm_weight[d]);
            out[idx] = qwen_bonsai2_round_mul_f32(
                normalized, qwen_bonsai2_silu_f32_exact(z[idx]));
        }
    }
    return 0;
}

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

static inline int32_t qwen_bonsai2_ptq1_dot_trits16_avx2(
        const uint8_t *packed, uint16_t pow3, const int8_t *activation) {
    const __m128i raw8 = _mm_loadu_si128((const __m128i *)packed);
    __m256i v = _mm256_cvtepu8_epi16(raw8);
    v = _mm256_mullo_epi16(v, _mm256_set1_epi16((short)pow3));
    v = _mm256_and_si256(v, _mm256_set1_epi16(0x00ff));
    v = _mm256_mullo_epi16(v, _mm256_set1_epi16(3));
    __m256i q = _mm256_srli_epi16(v, 8);
    q = _mm256_sub_epi16(q, _mm256_set1_epi16(1));

    const __m128i a8 = _mm_loadu_si128((const __m128i *)activation);
    const __m256i a16 = _mm256_cvtepi8_epi16(a8);
    const __m256i prod = _mm256_mullo_epi16(q, a16);
    const __m256i sum32 = _mm256_madd_epi16(
        prod, _mm256_set1_epi16(1));
    return qwen_hsum8_i32(sum32);
}

static inline int32_t qwen_bonsai2_ptq1_dot_trits8_sse(
        const uint8_t *packed, uint16_t pow3, const int8_t *activation) {
    const __m128i raw8 = _mm_loadl_epi64((const __m128i *)packed);
    __m128i v = _mm_cvtepu8_epi16(raw8);
    v = _mm_mullo_epi16(v, _mm_set1_epi16((short)pow3));
    v = _mm_and_si128(v, _mm_set1_epi16(0x00ff));
    v = _mm_mullo_epi16(v, _mm_set1_epi16(3));
    __m128i q = _mm_srli_epi16(v, 8);
    q = _mm_sub_epi16(q, _mm_set1_epi16(1));

    const __m128i a8 = _mm_loadl_epi64((const __m128i *)activation);
    const __m128i a16 = _mm_cvtepi8_epi16(a8);
    const __m128i prod = _mm_mullo_epi16(q, a16);
    __m128i sum32 = _mm_madd_epi16(prod, _mm_set1_epi16(1));
    sum32 = _mm_hadd_epi32(sum32, sum32);
    sum32 = _mm_hadd_epi32(sum32, sum32);
    return _mm_cvtsi128_si32(sum32);
}

/* Fused PTQ1 decoder + Q8 dot.
 *
 * PTQ1's 24 qs bytes decode in two Prism stages: 16 bytes x 5 trits
 * followed by 8 bytes x 5 trits, then 2 qh bytes x 4 trits. Decode each
 * contiguous output segment directly into the matching Q8_0 activation
 * segment. This removes the 128-byte q[] materialization and the second pass
 * over it while preserving the four integer dot sums and float accumulation
 * order of qwen_bonsai2_vec_dot_ptq1_q8_0 exactly. */
static float qwen_bonsai2_vec_dot_ptq1_q8_0_fused(
        const uint8_t *weights, const uint8_t *activation, size_t n,
        const int8_t lut[256][5]) {
    static const uint16_t pow3[5] = {1, 3, 9, 27, 81};
    const size_t nb = n / QWEN_QK_PTQ1_0;
    float sumf = 0.0f;

    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = weights + ib * QWEN_BLOCK_PTQ1_0;
        const uint8_t *qs = xb;
        const uint8_t *qh = xb + 24;
        const uint8_t *yb = activation + ib * 4 * QWEN_BLOCK_Q8_0;
        const int8_t *a0 = (const int8_t *)(yb + 0 * QWEN_BLOCK_Q8_0 + 2);
        const int8_t *a1 = (const int8_t *)(yb + 1 * QWEN_BLOCK_Q8_0 + 2);
        const int8_t *a2 = (const int8_t *)(yb + 2 * QWEN_BLOCK_Q8_0 + 2);
        const int8_t *a3 = (const int8_t *)(yb + 3 * QWEN_BLOCK_Q8_0 + 2);

        int32_t dots[4] = {0, 0, 0, 0};

        dots[0] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[0], a0 + 0);
        dots[0] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[1], a0 + 16);
        dots[1] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[2], a1 + 0);
        dots[1] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[3], a1 + 16);
        dots[2] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[4], a2 + 0);

        dots[2] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[0], a2 + 16);
        dots[2] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[1], a2 + 24);
        dots[3] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[2], a3 + 0);
        dots[3] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[3], a3 + 8);
        dots[3] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[4], a3 + 16);

        for (int nn = 0; nn < 4; ++nn) {
            for (int h = 0; h < 2; ++h) {
                dots[3] +=
                    (int32_t)lut[qh[h]][nn] *
                    (int32_t)a3[24 + nn * 2 + h];
            }
        }

        const float d0 = qwen_f16_to_f32(qwen_load_u16_le(xb + 26));
        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const uint8_t *ab = yb + k * QWEN_BLOCK_Q8_0;
            const float d1 = qwen_f16_to_f32(qwen_load_u16_le(ab));
            sumi += d1 * (float)dots[k];
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

/* Experimental exact PTQ1 path that decodes the shared Q8_0 activation
 * scales once per matvec rather than once per output-row/block.  The integer
 * dot products and all floating-point accumulation order remain identical to
 * qwen_bonsai2_vec_dot_ptq1_q8_0_fused(). */
static float qwen_bonsai2_vec_dot_ptq1_q8_0_fused_cached_scales(
        const uint8_t *weights,
        const uint8_t *activation,
        const float *activation_scales,
        size_t n,
        const int8_t lut[256][5]) {
    static const uint16_t pow3[5] = {1, 3, 9, 27, 81};
    const size_t nb = n / QWEN_QK_PTQ1_0;
    float sumf = 0.0f;

    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = weights + ib * QWEN_BLOCK_PTQ1_0;
        const uint8_t *qs = xb;
        const uint8_t *qh = xb + 24;
        const uint8_t *yb = activation + ib * 4 * QWEN_BLOCK_Q8_0;
        const int8_t *a0 = (const int8_t *)(yb + 0 * QWEN_BLOCK_Q8_0 + 2);
        const int8_t *a1 = (const int8_t *)(yb + 1 * QWEN_BLOCK_Q8_0 + 2);
        const int8_t *a2 = (const int8_t *)(yb + 2 * QWEN_BLOCK_Q8_0 + 2);
        const int8_t *a3 = (const int8_t *)(yb + 3 * QWEN_BLOCK_Q8_0 + 2);

        int32_t dots[4] = {0, 0, 0, 0};

        dots[0] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[0], a0 + 0);
        dots[0] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[1], a0 + 16);
        dots[1] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[2], a1 + 0);
        dots[1] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[3], a1 + 16);
        dots[2] += qwen_bonsai2_ptq1_dot_trits16_avx2(qs, pow3[4], a2 + 0);

        dots[2] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[0], a2 + 16);
        dots[2] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[1], a2 + 24);
        dots[3] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[2], a3 + 0);
        dots[3] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[3], a3 + 8);
        dots[3] += qwen_bonsai2_ptq1_dot_trits8_sse(qs + 16, pow3[4], a3 + 16);

        for (int nn = 0; nn < 4; ++nn) {
            for (int h = 0; h < 2; ++h) {
                dots[3] +=
                    (int32_t)lut[qh[h]][nn] *
                    (int32_t)a3[24 + nn * 2 + h];
            }
        }

        const float d0 = qwen_f16_to_f32(qwen_load_u16_le(xb + 26));
        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const float d1 = activation_scales[ib * 4 + (size_t)k];
            sumi += d1 * (float)dots[k];
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

static int qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 ||
        n % QWEN_QK_PTQ1_0 != 0) return -1;
    const size_t wr = (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t q8_blocks = n / QWEN_QK8_0;
    const size_t ar = q8_blocks * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar) return -2;
    if (weights_bytes != rows * wr) return -3;

    float *activation_scales =
        (float *)malloc(q8_blocks * sizeof(float));
    if (!activation_scales) return -4;
    for (size_t ib = 0; ib < q8_blocks; ++ib) {
        const uint8_t *ab = activation + ib * QWEN_BLOCK_Q8_0;
        activation_scales[ib] =
            qwen_f16_to_f32(qwen_load_u16_le(ab));
    }

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    for (size_t r = 0; r < rows; ++r) {
        out[r] = qwen_bonsai2_vec_dot_ptq1_q8_0_fused_cached_scales(
            weights + r * wr, activation, activation_scales, n, lut);
    }

    free(activation_scales);
    return 0;
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
    return qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
        weights, weights_bytes, rows, n,
        activation, activation_bytes, out);
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
