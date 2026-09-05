/* OpenMP row-parallel overlay for the proven scalar Qwen3.8 quant bridge.
 *
 * The scalar dot products and activation quantizers remain byte-for-byte the
 * existing implementation.  Only independent output rows are distributed
 * across OpenMP workers, so the reduction order inside every output row is
 * unchanged.  This file intentionally lives beside, rather than replaces,
 * gguf_quant_dot.c so the historical correctness lane remains available as a
 * serial A/B baseline.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define qwen_matvec_q8_0_q8_0_scalar qwen_matvec_q8_0_q8_0_serial
#define qwen_matvec_q6_k_q8_k_scalar qwen_matvec_q6_k_q8_k_serial
#include "gguf_quant_dot.c"
#undef qwen_matvec_q8_0_q8_0_scalar
#undef qwen_matvec_q6_k_q8_k_scalar

#ifdef _OPENMP
#include <omp.h>
#endif

int qwen_quant_parallel_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int qwen_matvec_q8_0_q8_0_scalar(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t row_bytes = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    if (activation_bytes != row_bytes) return -2;
    if (weights_bytes % row_bytes != 0 || weights_bytes / row_bytes != rows) return -3;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t row = 0; row < (ptrdiff_t) rows; ++row) {
        out[row] = qwen_vec_dot_q8_0_q8_0_scalar(
            weights + (size_t) row * row_bytes,
            row_bytes,
            activation,
            activation_bytes,
            n);
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

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t row = 0; row < (ptrdiff_t) rows; ++row) {
        out[row] = qwen_vec_dot_q6_k_q8_k_scalar(
            weights + (size_t) row * weight_row_bytes,
            weight_row_bytes,
            activation,
            activation_bytes,
            n);
    }
    return 0;
}

#ifdef QWEN_QUANT_PARALLEL_SELFTEST
static int same_f32_bits(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

int main(void) {
    enum { N = 256, ROWS = 7 };
    float x[N];
    for (int i = 0; i < N; ++i) x[i] = (float) (((i * 37) % 251) - 125) / 91.0f;

    uint8_t q8a[QWEN_BLOCK_Q8_0 * (N / QWEN_QK8_0)];
    if (qwen_quantize_q8_0_scalar(x, N, q8a, sizeof(q8a)) != 0) return 2;
    uint8_t q8w[ROWS * sizeof(q8a)];
    for (int r = 0; r < ROWS; ++r) {
        float w[N];
        for (int i = 0; i < N; ++i) w[i] = (float) ((((i + 11 * r) * 29) % 257) - 128) / 73.0f;
        if (qwen_quantize_q8_0_scalar(w, N, q8w + (size_t) r * sizeof(q8a), sizeof(q8a)) != 0) return 3;
    }
    float q8_serial[ROWS], q8_parallel[ROWS];
    if (qwen_matvec_q8_0_q8_0_serial(q8w, sizeof(q8w), ROWS, N, q8a, sizeof(q8a), q8_serial) != 0) return 4;
    if (qwen_matvec_q8_0_q8_0_scalar(q8w, sizeof(q8w), ROWS, N, q8a, sizeof(q8a), q8_parallel) != 0) return 5;
    if (!same_f32_bits(q8_serial, q8_parallel, ROWS)) return 6;

    uint8_t q8k[QWEN_BLOCK_Q8_K];
    if (qwen_quantize_q8_k_scalar(x, N, q8k, sizeof(q8k)) != 0) return 7;
    uint8_t q6w[ROWS * QWEN_BLOCK_Q6_K];
    for (int r = 0; r < ROWS; ++r) {
        uint8_t *row = q6w + (size_t) r * QWEN_BLOCK_Q6_K;
        for (int i = 0; i < 128; ++i) row[i] = (uint8_t) ((i * 17 + r * 13) & 0xff);
        for (int i = 0; i < 64; ++i) row[128 + i] = (uint8_t) ((i * 23 + r * 7) & 0xff);
        for (int i = 0; i < 16; ++i) row[192 + i] = (uint8_t) (int8_t) ((i % 2 ? 1 : -1) * (i + 1));
        /* finite FP16 super-scale: 1.0 */
        row[208] = 0x00;
        row[209] = 0x3c;
    }
    float q6_serial[ROWS], q6_parallel[ROWS];
    if (qwen_matvec_q6_k_q8_k_serial(q6w, sizeof(q6w), ROWS, N, q8k, sizeof(q8k), q6_serial) != 0) return 8;
    if (qwen_matvec_q6_k_q8_k_scalar(q6w, sizeof(q6w), ROWS, N, q8k, sizeof(q8k), q6_parallel) != 0) return 9;
    if (!same_f32_bits(q6_serial, q6_parallel, ROWS)) return 10;

    printf("QWEN_QUANT_OPENMP_ROW_EXACT_PASS threads=%d rows=%d\n",
           qwen_quant_parallel_max_threads(), ROWS);
    return 0;
}
#endif
