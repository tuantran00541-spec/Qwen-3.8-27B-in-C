/* Exact Q8_0 matvec-many allocation-hoist candidate for Qwen3.8.
 *
 * The proven baseline allocates/free()s the n_vec F32 accumulation buffer once
 * per output row.  This candidate allocates it once per matvec call and zeros
 * it before each row.  Weight traversal, integer dot products, scale loads,
 * F32 multiply/add order, and output layout are intentionally unchanged.
 *
 * Q6_K delegates byte-for-byte to the already-proven exact many-vector kernel.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gguf_quant_matvec_many_avx2.c"

static int qwen_matvec_many_q8_0_q8_0_noalloc_exact(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    if (!weights || !activations || !out || rows == 0 || n_vec == 0 || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t nb = n / QWEN_QK8_0;
    const size_t row_bytes = nb * QWEN_BLOCK_Q8_0;
    if (activation_bytes_each != row_bytes) return -2;
    if (weights_bytes != rows * row_bytes) return -3;

    float *sums = (float *) calloc(n_vec, sizeof(float));
    if (!sums) return -4;

    for (size_t r = 0; r < rows; ++r) {
        memset(sums, 0, n_vec * sizeof(float));
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
    }

    free(sums);
    return 0;
}

int qwen_matvec_many_q8_0_q8_0_bridge(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    return qwen_matvec_many_q8_0_q8_0_noalloc_exact(
        weights, weights_bytes, rows, n,
        activations, activation_bytes_each, n_vec, out);
}

int qwen_matvec_many_q6_k_q8_k_bridge(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    return qwen_matvec_many_q6_k_q8_k_exact(
        weights, weights_bytes, rows, n,
        activations, activation_bytes_each, n_vec, out);
}
