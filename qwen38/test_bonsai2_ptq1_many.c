/* RED/GREEN bitwise + same-run benchmark gate for PTQ1 matvec-many.
 *
 * Sequential reference calls the CURRENT production PTQ1 matvec once per
 * activation vector.  Candidate must traverse/decode each PTQ1 weight block
 * once and apply it to 2/4/8 already-quantized Q8_0 activations while keeping
 * each vector's floating-point accumulation order bitwise identical.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bonsai2_quant_dot_avx2.c"

static uint32_t rng_state = 0x1355a11du;
static volatile float sink_value = 0.0f;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static uint16_t random_f16_scale(void) {
    static const uint16_t scales[] = {
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125  */
        0x3400u, /* 0.25   */
        0x3800u, /* 0.5    */
        0x3c00u, /* 1.0    */
    };
    return scales[rng_u32() % (sizeof(scales) / sizeof(scales[0]))];
}

static double now_seconds(void) {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_ptq1(uint8_t *weights, size_t rows, size_t n) {
    const size_t nb = n / QWEN_QK_PTQ1_0;
    const size_t row_bytes = nb * QWEN_BLOCK_PTQ1_0;
    for (size_t r = 0; r < rows; ++r) {
        uint8_t *row = weights + r * row_bytes;
        for (size_t ib = 0; ib < nb; ++ib) {
            uint8_t *b = row + ib * QWEN_BLOCK_PTQ1_0;
            for (int i = 0; i < 26; ++i) b[i] = (uint8_t)rng_u32();
            const uint16_t d = random_f16_scale();
            b[26] = (uint8_t)(d & 0xffu);
            b[27] = (uint8_t)(d >> 8);
        }
    }
}

static void fill_q8(uint8_t *activation, size_t n) {
    const size_t nb = n / QWEN_QK8_0;
    for (size_t ib = 0; ib < nb; ++ib) {
        uint8_t *b = activation + ib * QWEN_BLOCK_Q8_0;
        const uint16_t d = random_f16_scale();
        b[0] = (uint8_t)(d & 0xffu);
        b[1] = (uint8_t)(d >> 8);
        for (int i = 0; i < 32; ++i) {
            b[2 + i] = (uint8_t)((int)(rng_u32() % 255u) - 127);
        }
    }
}

static int exact_case(size_t rows, size_t n, size_t n_vec) {
    const size_t wr = (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ar = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc(rows * wr);
    uint8_t *acts = (uint8_t *)malloc(n_vec * ar);
    float *reference = (float *)malloc(n_vec * rows * sizeof(float));
    float *many = (float *)malloc(n_vec * rows * sizeof(float));
    if (!weights || !acts || !reference || !many) return 90;

    fill_ptq1(weights, rows, n);
    for (size_t v = 0; v < n_vec; ++v) fill_q8(acts + v * ar, n);

    for (size_t v = 0; v < n_vec; ++v) {
        const int rc = qwen_bonsai2_matvec_ptq1_0_q8_0(
            weights, rows * wr, rows, n,
            acts + v * ar, ar,
            reference + v * rows);
        if (rc != 0) return 91;
    }

    const int rc = qwen_bonsai2_matvec_many_ptq1_0_q8_0(
        weights, rows * wr, rows, n,
        acts, ar, n_vec, many);
    if (rc != 0) {
        fprintf(stderr, "PTQ1 many rc=%d rows=%zu n=%zu nv=%zu\n",
                rc, rows, n, n_vec);
        return 92;
    }

    if (memcmp(reference, many, n_vec * rows * sizeof(float)) != 0) {
        for (size_t i = 0; i < n_vec * rows; ++i) {
            uint32_t a, b;
            memcpy(&a, reference + i, sizeof(a));
            memcpy(&b, many + i, sizeof(b));
            if (a != b) {
                fprintf(stderr,
                        "PTQ1 many mismatch nv=%zu index=%zu ref=%08x got=%08x\n",
                        n_vec, i, a, b);
                break;
            }
        }
        return 1;
    }

    free(weights);
    free(acts);
    free(reference);
    free(many);
    return 0;
}

static int bench_case(size_t n_vec) {
    const size_t rows = 2048;
    const size_t n = 5120;
    const int repeats = 6;
    const size_t wr = (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ar = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;

    uint8_t *weights = (uint8_t *)malloc(rows * wr);
    uint8_t *acts = (uint8_t *)malloc(n_vec * ar);
    float *out = (float *)malloc(n_vec * rows * sizeof(float));
    if (!weights || !acts || !out) return 93;

    fill_ptq1(weights, rows, n);
    for (size_t v = 0; v < n_vec; ++v) fill_q8(acts + v * ar, n);

    double t0 = now_seconds();
    float seq_sink = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        for (size_t v = 0; v < n_vec; ++v) {
            const int rc = qwen_bonsai2_matvec_ptq1_0_q8_0(
                weights, rows * wr, rows, n,
                acts + v * ar, ar,
                out + v * rows);
            if (rc != 0) return 94;
        }
        seq_sink += out[(size_t)rep % (n_vec * rows)];
    }
    const double sequential_s = now_seconds() - t0;

    t0 = now_seconds();
    float many_sink = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        const int rc = qwen_bonsai2_matvec_many_ptq1_0_q8_0(
            weights, rows * wr, rows, n,
            acts, ar, n_vec, out);
        if (rc != 0) return 95;
        many_sink += out[(size_t)rep % (n_vec * rows)];
    }
    const double many_s = now_seconds() - t0;
    sink_value = seq_sink + many_sink;

    printf("QWEN38_BONSAI2_PTQ1_MANY_BENCH "
           "nv=%zu rows=%zu n=%zu repeats=%d "
           "sequential_seconds=%.9f many_seconds=%.9f speedup=%.4fx "
           "effective_vectors_per_second_seq=%.3f "
           "effective_vectors_per_second_many=%.3f\n",
           n_vec, rows, n, repeats,
           sequential_s, many_s, sequential_s / many_s,
           (double)(repeats * n_vec) / sequential_s,
           (double)(repeats * n_vec) / many_s);

    free(weights);
    free(acts);
    free(out);
    return 0;
}

int main(void) {
    static const size_t nvecs[] = {2, 4, 8};
    static const size_t widths[] = {128, 1024, 5120, 17408};

    for (size_t nv_i = 0; nv_i < sizeof(nvecs)/sizeof(nvecs[0]); ++nv_i) {
        const size_t nv = nvecs[nv_i];
        for (size_t wi = 0; wi < sizeof(widths)/sizeof(widths[0]); ++wi) {
            const int rc = exact_case(31, widths[wi], nv);
            if (rc != 0) return rc;
        }
    }
    puts("QWEN38_BONSAI2_PTQ1_MANY_BITWISE_PASS");

    for (size_t nv_i = 0; nv_i < sizeof(nvecs)/sizeof(nvecs[0]); ++nv_i) {
        const int rc = bench_case(nvecs[nv_i]);
        if (rc != 0) return rc;
    }
    return 0;
}
