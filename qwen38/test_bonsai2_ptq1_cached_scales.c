#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bonsai2_quant_dot_avx2.c"

static uint32_t rng_state = 0x5ca1e5u;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static uint16_t fixture_scale(void) {
    static const float values[] = {
        0.03125f, 0.0625f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f
    };
    return qwen_f32_to_f16(values[rng_u32() % 7u]);
}

static void fill_fixture(
        uint8_t *weights,
        size_t rows,
        size_t n,
        uint8_t *activation) {
    const size_t nb = n / QWEN_QK_PTQ1_0;
    const size_t wr = nb * QWEN_BLOCK_PTQ1_0;
    const size_t ab = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;

    for (size_t r = 0; r < rows; ++r) {
        uint8_t *row = weights + r * wr;
        for (size_t ib = 0; ib < nb; ++ib) {
            uint8_t *block = row + ib * QWEN_BLOCK_PTQ1_0;
            for (int i = 0; i < 26; ++i) {
                block[i] = (uint8_t)rng_u32();
            }
            const uint16_t d = fixture_scale();
            block[26] = (uint8_t)(d & 0xffu);
            block[27] = (uint8_t)(d >> 8);
        }
    }

    for (size_t off = 0; off < ab; off += QWEN_BLOCK_Q8_0) {
        const uint16_t d = fixture_scale();
        activation[off + 0] = (uint8_t)(d & 0xffu);
        activation[off + 1] = (uint8_t)(d >> 8);
        for (int i = 0; i < 32; ++i) {
            activation[off + 2 + i] =
                (uint8_t)((int)(rng_u32() % 255u) - 127);
        }
    }
}

static int check_case(size_t rows, size_t n) {
    const size_t wr = (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ab = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc(rows * wr);
    uint8_t *activation = (uint8_t *)malloc(ab);
    float *reference = (float *)malloc(rows * sizeof(float));
    float *candidate = (float *)malloc(rows * sizeof(float));
    if (!weights || !activation || !reference || !candidate) return 90;

    fill_fixture(weights, rows, n, activation);
    if (qwen_bonsai2_matvec_ptq1_0_q8_0(
            weights, rows * wr, rows, n,
            activation, ab, reference) != 0) return 91;
    if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
            weights, rows * wr, rows, n,
            activation, ab, candidate) != 0) return 92;

    const int same = memcmp(reference, candidate, rows * sizeof(float)) == 0;
    if (!same) {
        for (size_t r = 0; r < rows; ++r) {
            uint32_t a, b;
            memcpy(&a, &reference[r], sizeof(a));
            memcpy(&b, &candidate[r], sizeof(b));
            if (a != b) {
                fprintf(stderr,
                        "cached-scale mismatch rows=%zu n=%zu row=%zu "
                        "ref=%.9g cand=%.9g ref_bits=%08x cand_bits=%08x\n",
                        rows, n, r, reference[r], candidate[r], a, b);
                break;
            }
        }
    }

    free(weights);
    free(activation);
    free(reference);
    free(candidate);
    return same ? 0 : 93;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int benchmark(void) {
    enum { ROWS = 4096, N = 5120, REPS = 8 };
    const size_t wr = (N / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ab = (N / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc((size_t)ROWS * wr);
    uint8_t *activation = (uint8_t *)malloc(ab);
    float *out = (float *)malloc((size_t)ROWS * sizeof(float));
    if (!weights || !activation || !out) return 94;
    fill_fixture(weights, ROWS, N, activation);

    if (qwen_bonsai2_matvec_ptq1_0_q8_0(
            weights, (size_t)ROWS * wr, ROWS, N,
            activation, ab, out) != 0) return 95;
    if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
            weights, (size_t)ROWS * wr, ROWS, N,
            activation, ab, out) != 0) return 96;

    double t0 = now_s();
    for (int i = 0; i < REPS; ++i) {
        if (qwen_bonsai2_matvec_ptq1_0_q8_0(
                weights, (size_t)ROWS * wr, ROWS, N,
                activation, ab, out) != 0) return 97;
    }
    const double reference_s = now_s() - t0;

    t0 = now_s();
    for (int i = 0; i < REPS; ++i) {
        if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
                weights, (size_t)ROWS * wr, ROWS, N,
                activation, ab, out) != 0) return 98;
    }
    const double candidate_s = now_s() - t0;

    printf(
        "PTQ1 cached activation scales reference=%.6f candidate=%.6f speedup=%.4fx\n",
        reference_s, candidate_s,
        candidate_s > 0.0 ? reference_s / candidate_s : 0.0);

    free(weights);
    free(activation);
    free(out);
    return 0;
}

int main(void) {
    if (check_case(257, 128) != 0) return 1;
    if (check_case(257, 5120) != 0) return 2;
    if (check_case(17, 17408) != 0) return 3;
    if (benchmark() != 0) return 4;
    puts("QWEN38_BONSAI2_PTQ1_CACHED_SCALES_BITWISE_PASS");
    return 0;
}
