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

static int reference_matvec(
        const uint8_t *weights,
        size_t weights_bytes,
        size_t rows,
        size_t n,
        const uint8_t *activation,
        size_t activation_bytes,
        float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 ||
        n % QWEN_QK_PTQ1_0 != 0) return -1;
    const size_t wr = (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ar = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar || weights_bytes != rows * wr) return -2;

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    for (size_t r = 0; r < rows; ++r) {
        out[r] = qwen_bonsai2_vec_dot_ptq1_q8_0_fused(
            weights + r * wr, activation, n, lut);
    }
    return 0;
}


static int candidate_matvec_rows2(
        const uint8_t *weights,
        size_t weights_bytes,
        size_t rows,
        size_t n,
        const uint8_t *activation,
        size_t activation_bytes,
        float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 ||
        n % QWEN_QK_PTQ1_0 != 0) return -1;
    const size_t nb = n / QWEN_QK_PTQ1_0;
    const size_t wr = nb * QWEN_BLOCK_PTQ1_0;
    const size_t q8_blocks = n / QWEN_QK8_0;
    const size_t ar = q8_blocks * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar || weights_bytes != rows * wr) return -2;

    float *activation_scales =
        (float *)malloc(q8_blocks * sizeof(float));
    if (!activation_scales) return -3;
    for (size_t ib = 0; ib < q8_blocks; ++ib) {
        activation_scales[ib] = qwen_f16_to_f32(
            qwen_load_u16_le(
                activation + ib * QWEN_BLOCK_Q8_0));
    }

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);

    size_t r = 0;
    for (; r + 1 < rows; r += 2) {
        const uint8_t *row0 = weights + (r + 0) * wr;
        const uint8_t *row1 = weights + (r + 1) * wr;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (size_t ib = 0; ib < nb; ++ib) {
            const uint8_t *yb =
                activation + ib * 4 * QWEN_BLOCK_Q8_0;
            const float *scales = activation_scales + ib * 4;

            int32_t dots0[4];
            const uint8_t *xb0 =
                row0 + ib * QWEN_BLOCK_PTQ1_0;
            qwen_bonsai2_ptq1_dot_block_reuse_avx2(
                xb0, yb, lut, dots0);
            const float d00 =
                qwen_f16_to_f32(qwen_load_u16_le(xb0 + 26));
            float sumi0 = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sumi0 += scales[k] * (float)dots0[k];
            }
            sum0 += d00 * sumi0;

            int32_t dots1[4];
            const uint8_t *xb1 =
                row1 + ib * QWEN_BLOCK_PTQ1_0;
            qwen_bonsai2_ptq1_dot_block_reuse_avx2(
                xb1, yb, lut, dots1);
            const float d01 =
                qwen_f16_to_f32(qwen_load_u16_le(xb1 + 26));
            float sumi1 = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sumi1 += scales[k] * (float)dots1[k];
            }
            sum1 += d01 * sumi1;
        }
        out[r + 0] = sum0;
        out[r + 1] = sum1;
    }

    if (r < rows) {
        out[r] = qwen_bonsai2_vec_dot_ptq1_q8_0_fused_cached_scales(
            weights + r * wr, activation, activation_scales, n, lut);
    }

    free(activation_scales);
    return 0;
}

static int candidate_matvec_rows4(
        const uint8_t *weights,
        size_t weights_bytes,
        size_t rows,
        size_t n,
        const uint8_t *activation,
        size_t activation_bytes,
        float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 ||
        n % QWEN_QK_PTQ1_0 != 0) return -1;
    const size_t nb = n / QWEN_QK_PTQ1_0;
    const size_t wr = nb * QWEN_BLOCK_PTQ1_0;
    const size_t q8_blocks = n / QWEN_QK8_0;
    const size_t ar = q8_blocks * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar || weights_bytes != rows * wr) return -2;

    float *activation_scales =
        (float *)malloc(q8_blocks * sizeof(float));
    if (!activation_scales) return -3;
    for (size_t ib = 0; ib < q8_blocks; ++ib) {
        activation_scales[ib] = qwen_f16_to_f32(
            qwen_load_u16_le(
                activation + ib * QWEN_BLOCK_Q8_0));
    }

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);

    size_t r = 0;
    for (; r + 3 < rows; r += 4) {
        const uint8_t *row0 = weights + (r + 0) * wr;
        const uint8_t *row1 = weights + (r + 1) * wr;
        const uint8_t *row2 = weights + (r + 2) * wr;
        const uint8_t *row3 = weights + (r + 3) * wr;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        for (size_t ib = 0; ib < nb; ++ib) {
            const uint8_t *yb =
                activation + ib * 4 * QWEN_BLOCK_Q8_0;
            const float *scales = activation_scales + ib * 4;

#define QWEN_ROW4_STEP(ROWPTR, SUMVAR) do { \
            const uint8_t *xb = \
                (ROWPTR) + ib * QWEN_BLOCK_PTQ1_0; \
            int32_t dots[4]; \
            qwen_bonsai2_ptq1_dot_block_reuse_avx2( \
                xb, yb, lut, dots); \
            const float d0 = \
                qwen_f16_to_f32(qwen_load_u16_le(xb + 26)); \
            float sumi = 0.0f; \
            for (int k = 0; k < 4; ++k) { \
                sumi += scales[k] * (float)dots[k]; \
            } \
            (SUMVAR) += d0 * sumi; \
        } while (0)

            QWEN_ROW4_STEP(row0, sum0);
            QWEN_ROW4_STEP(row1, sum1);
            QWEN_ROW4_STEP(row2, sum2);
            QWEN_ROW4_STEP(row3, sum3);
#undef QWEN_ROW4_STEP
        }

        out[r + 0] = sum0;
        out[r + 1] = sum1;
        out[r + 2] = sum2;
        out[r + 3] = sum3;
    }

    for (; r < rows; ++r) {
        out[r] = qwen_bonsai2_vec_dot_ptq1_q8_0_fused_cached_scales(
            weights + r * wr, activation, activation_scales, n, lut);
    }

    free(activation_scales);
    return 0;
}

static int check_interleaved_case(size_t rows, size_t n) {
    const size_t wr = (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ab = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc(rows * wr);
    uint8_t *activation = (uint8_t *)malloc(ab);
    float *production = (float *)malloc(rows * sizeof(float));
    float *rows2 = (float *)malloc(rows * sizeof(float));
    float *rows4 = (float *)malloc(rows * sizeof(float));
    if (!weights || !activation || !production || !rows2 || !rows4) {
        free(rows4);
        free(rows2);
        free(production);
        free(activation);
        free(weights);
        return 80;
    }

    fill_fixture(weights, rows, n, activation);
    if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
            weights, rows * wr, rows, n,
            activation, ab, production) != 0) return 81;
    if (candidate_matvec_rows2(
            weights, rows * wr, rows, n,
            activation, ab, rows2) != 0) return 82;
    if (candidate_matvec_rows4(
            weights, rows * wr, rows, n,
            activation, ab, rows4) != 0) return 83;

    const int same2 =
        memcmp(production, rows2, rows * sizeof(float)) == 0;
    const int same4 =
        memcmp(production, rows4, rows * sizeof(float)) == 0;
    if (!same2 || !same4) {
        fprintf(stderr,
                "row-interleave mismatch rows=%zu n=%zu rows2=%d rows4=%d\n",
                rows, n, same2, same4);
    }

    free(rows4);
    free(rows2);
    free(production);
    free(activation);
    free(weights);
    return same2 && same4 ? 0 : 84;
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
    if (reference_matvec(
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
    enum { ROWS = 4096, N = 5120, SAMPLES = 6 };
    const size_t wr = (N / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ab = (N / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc((size_t)ROWS * wr);
    uint8_t *activation = (uint8_t *)malloc(ab);
    float *out = (float *)malloc((size_t)ROWS * sizeof(float));
    if (!weights || !activation || !out) return 94;
    fill_fixture(weights, ROWS, N, activation);

    if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
            weights, (size_t)ROWS * wr, ROWS, N,
            activation, ab, out) != 0) return 95;
    if (candidate_matvec_rows2(
            weights, (size_t)ROWS * wr, ROWS, N,
            activation, ab, out) != 0) return 96;
    if (candidate_matvec_rows4(
            weights, (size_t)ROWS * wr, ROWS, N,
            activation, ab, out) != 0) return 97;

    double production[SAMPLES];
    double rows2[SAMPLES];
    double rows4[SAMPLES];

    for (int sample = 0; sample < SAMPLES; ++sample) {
        for (int slot = 0; slot < 3; ++slot) {
            const int which = (sample + slot) % 3;
            const double t0 = now_s();
            int rc;
            if (which == 0) {
                rc = qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
                    weights, (size_t)ROWS * wr, ROWS, N,
                    activation, ab, out);
            } else if (which == 1) {
                rc = candidate_matvec_rows2(
                    weights, (size_t)ROWS * wr, ROWS, N,
                    activation, ab, out);
            } else {
                rc = candidate_matvec_rows4(
                    weights, (size_t)ROWS * wr, ROWS, N,
                    activation, ab, out);
            }
            const double elapsed = now_s() - t0;
            if (rc != 0) return 98 + which;
            if (which == 0) production[sample] = elapsed;
            if (which == 1) rows2[sample] = elapsed;
            if (which == 2) rows4[sample] = elapsed;
        }
    }

    double prod_sum = 0.0;
    double rows2_sum = 0.0;
    double rows4_sum = 0.0;
    for (int i = 0; i < SAMPLES; ++i) {
        prod_sum += production[i];
        rows2_sum += rows2[i];
        rows4_sum += rows4[i];
    }
    const double prod_mean = prod_sum / (double)SAMPLES;
    const double rows2_mean = rows2_sum / (double)SAMPLES;
    const double rows4_mean = rows4_sum / (double)SAMPLES;

    printf(
        "QWEN38_BONSAI2_PTQ1_ROW_INTERLEAVE_BENCH "
        "rows=%d n=%d samples=%d "
        "production_mean_seconds=%.9f "
        "rows2_mean_seconds=%.9f rows2_speedup=%.4fx "
        "rows4_mean_seconds=%.9f rows4_speedup=%.4fx\n",
        ROWS, N, SAMPLES,
        prod_mean,
        rows2_mean, rows2_mean > 0.0 ? prod_mean / rows2_mean : 0.0,
        rows4_mean, rows4_mean > 0.0 ? prod_mean / rows4_mean : 0.0);

    printf(
        "QWEN38_BONSAI2_PTQ1_ROW_INTERLEAVE_SAMPLES "
        "production=[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f] "
        "rows2=[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f] "
        "rows4=[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f]\n",
        production[0], production[1], production[2],
        production[3], production[4], production[5],
        rows2[0], rows2[1], rows2[2],
        rows2[3], rows2[4], rows2[5],
        rows4[0], rows4[1], rows4[2],
        rows4[3], rows4[4], rows4[5]);

    free(weights);
    free(activation);
    free(out);
    return 0;
}

int main(void) {
    if (check_case(257, 128) != 0) return 1;
    if (check_case(257, 5120) != 0) return 2;
    if (check_case(17, 17408) != 0) return 3;
    if (check_interleaved_case(257, 128) != 0) return 4;
    if (check_interleaved_case(257, 5120) != 0) return 5;
    if (check_interleaved_case(17, 17408) != 0) return 6;
    if (benchmark() != 0) return 7;
    puts("QWEN38_BONSAI2_PTQ1_ROW_INTERLEAVE_BITWISE_PASS");
    puts("QWEN38_BONSAI2_PTQ1_CACHED_SCALES_BITWISE_PASS");
    return 0;
}
