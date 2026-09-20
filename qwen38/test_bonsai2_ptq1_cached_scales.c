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


static inline void scalar_qh_block_reuse(
        const uint8_t *xb,
        const uint8_t *yb,
        const int8_t lut[256][5],
        int32_t dots[4]) {
    static const uint16_t pow3[5] = {1, 3, 9, 27, 81};
    const uint8_t *qs = xb;
    const uint8_t *qh = xb + 24;
    const int8_t *a0 =
        (const int8_t *)(yb + 0 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a1 =
        (const int8_t *)(yb + 1 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a2 =
        (const int8_t *)(yb + 2 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a3 =
        (const int8_t *)(yb + 3 * QWEN_BLOCK_Q8_0 + 2);

    const __m128i packed16 =
        _mm_loadu_si128((const __m128i *)qs);
    const __m256i raw16 =
        _mm256_cvtepu8_epi16(packed16);
    const __m128i packed8 =
        _mm_loadl_epi64((const __m128i *)(qs + 16));
    const __m128i raw8 =
        _mm_cvtepu8_epi16(packed8);

    const __m256i d00 =
        qwen_bonsai2_ptq1_pairs16_from_raw_avx2(
            raw16, pow3[0], a0 + 0);
    const __m256i d01 =
        qwen_bonsai2_ptq1_pairs16_from_raw_avx2(
            raw16, pow3[1], a0 + 16);
    dots[0] = qwen_hsum8_i32(
        _mm256_add_epi32(d00, d01));

    const __m256i d10 =
        qwen_bonsai2_ptq1_pairs16_from_raw_avx2(
            raw16, pow3[2], a1 + 0);
    const __m256i d11 =
        qwen_bonsai2_ptq1_pairs16_from_raw_avx2(
            raw16, pow3[3], a1 + 16);
    dots[1] = qwen_hsum8_i32(
        _mm256_add_epi32(d10, d11));

    const __m256i d20 =
        qwen_bonsai2_ptq1_pairs16_from_raw_avx2(
            raw16, pow3[4], a2 + 0);
    const __m128i d21 =
        qwen_bonsai2_ptq1_pairs8_from_raw_sse(
            raw8, pow3[0], a2 + 16);
    const __m128i d22 =
        qwen_bonsai2_ptq1_pairs8_from_raw_sse(
            raw8, pow3[1], a2 + 24);
    dots[2] =
        qwen_hsum8_i32(d20) +
        qwen_bonsai2_hsum4_i32_sse(
            _mm_add_epi32(d21, d22));

    const __m128i d30 =
        qwen_bonsai2_ptq1_pairs8_from_raw_sse(
            raw8, pow3[2], a3 + 0);
    const __m128i d31 =
        qwen_bonsai2_ptq1_pairs8_from_raw_sse(
            raw8, pow3[3], a3 + 8);
    const __m128i d32 =
        qwen_bonsai2_ptq1_pairs8_from_raw_sse(
            raw8, pow3[4], a3 + 16);
    dots[3] =
        qwen_bonsai2_hsum4_i32_sse(
            _mm_add_epi32(
                _mm_add_epi32(d30, d31), d32));

    for (int nn = 0; nn < 4; ++nn) {
        for (int hh = 0; hh < 2; ++hh) {
            dots[3] +=
                (int32_t)lut[qh[hh]][nn] *
                (int32_t)a3[24 + nn * 2 + hh];
        }
    }
}

static float scalar_qh_vec_dot_cached(
        const uint8_t *weights,
        const uint8_t *activation,
        const float *activation_scales,
        size_t n,
        const int8_t lut[256][5]) {
    const size_t nb = n / QWEN_QK_PTQ1_0;
    float sumf = 0.0f;
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb =
            weights + ib * QWEN_BLOCK_PTQ1_0;
        const uint8_t *yb =
            activation + ib * 4 * QWEN_BLOCK_Q8_0;
        int32_t dots[4];
        scalar_qh_block_reuse(xb, yb, lut, dots);

        const float d0 =
            qwen_f16_to_f32(qwen_load_u16_le(xb + 26));
        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            sumi +=
                activation_scales[ib * 4 + (size_t)k] *
                (float)dots[k];
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

static int scalar_qh_matvec_cached(
        const uint8_t *weights,
        size_t weights_bytes,
        size_t rows,
        size_t n,
        const uint8_t *activation,
        size_t activation_bytes,
        float *out) {
    if (!weights || !activation || !out ||
        rows == 0 || n == 0 ||
        n % QWEN_QK_PTQ1_0 != 0) return -1;
    const size_t wr =
        (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t q8_blocks = n / QWEN_QK8_0;
    const size_t ar = q8_blocks * QWEN_BLOCK_Q8_0;
    if (activation_bytes != ar ||
        weights_bytes != rows * wr) return -2;

    float *activation_scales =
        (float *)malloc(q8_blocks * sizeof(float));
    if (!activation_scales) return -3;
    for (size_t ib = 0; ib < q8_blocks; ++ib) {
        activation_scales[ib] =
            qwen_f16_to_f32(qwen_load_u16_le(
                activation + ib * QWEN_BLOCK_Q8_0));
    }

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    for (size_t r = 0; r < rows; ++r) {
        out[r] = scalar_qh_vec_dot_cached(
            weights + r * wr,
            activation,
            activation_scales,
            n,
            lut);
    }
    free(activation_scales);
    return 0;
}

static int check_qh_vector_case(size_t rows, size_t n) {
    const size_t wr =
        (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ab =
        (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc(rows * wr);
    uint8_t *activation = (uint8_t *)malloc(ab);
    float *scalar = (float *)malloc(rows * sizeof(float));
    float *vector = (float *)malloc(rows * sizeof(float));
    if (!weights || !activation || !scalar || !vector) return 70;

    fill_fixture(weights, rows, n, activation);
    if (scalar_qh_matvec_cached(
            weights, rows * wr, rows, n,
            activation, ab, scalar) != 0) return 71;
    if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
            weights, rows * wr, rows, n,
            activation, ab, vector) != 0) return 72;

    const int same =
        memcmp(scalar, vector, rows * sizeof(float)) == 0;
    if (!same) {
        fprintf(stderr,
                "qh-vector matvec mismatch rows=%zu n=%zu\n",
                rows, n);
    }
    free(vector);
    free(scalar);
    free(activation);
    free(weights);
    return same ? 0 : 73;
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
    enum { ROWS = 4096, N = 5120, REPS = 8 };
    const size_t wr = (N / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ab = (N / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc((size_t)ROWS * wr);
    uint8_t *activation = (uint8_t *)malloc(ab);
    float *out = (float *)malloc((size_t)ROWS * sizeof(float));
    if (!weights || !activation || !out) return 94;
    fill_fixture(weights, ROWS, N, activation);

    if (reference_matvec(
            weights, (size_t)ROWS * wr, ROWS, N,
            activation, ab, out) != 0) return 95;
    if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
            weights, (size_t)ROWS * wr, ROWS, N,
            activation, ab, out) != 0) return 96;

    double t0 = now_s();
    for (int i = 0; i < REPS; ++i) {
        if (reference_matvec(
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

static int benchmark_qh_geometry(
        size_t rows, size_t n, const char *label) {
    enum { SAMPLES = 4 };
    const size_t wr =
        (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t ab =
        (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *weights = (uint8_t *)malloc(rows * wr);
    uint8_t *activation = (uint8_t *)malloc(ab);
    float *out = (float *)malloc(rows * sizeof(float));
    if (!weights || !activation || !out) return 110;

    fill_fixture(weights, rows, n, activation);
    if (scalar_qh_matvec_cached(
            weights, rows * wr, rows, n,
            activation, ab, out) != 0) return 111;
    if (qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
            weights, rows * wr, rows, n,
            activation, ab, out) != 0) return 112;

    double scalar[SAMPLES];
    double vector[SAMPLES];
    for (int sample = 0; sample < SAMPLES; ++sample) {
        for (int slot = 0; slot < 2; ++slot) {
            const int which = (sample + slot) & 1;
            const double t0 = now_s();
            int rc;
            if (which == 0) {
                rc = scalar_qh_matvec_cached(
                    weights, rows * wr, rows, n,
                    activation, ab, out);
                scalar[sample] = now_s() - t0;
            } else {
                rc = qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(
                    weights, rows * wr, rows, n,
                    activation, ab, out);
                vector[sample] = now_s() - t0;
            }
            if (rc != 0) return 113 + which;
        }
    }

    double scalar_sum = 0.0;
    double vector_sum = 0.0;
    for (int i = 0; i < SAMPLES; ++i) {
        scalar_sum += scalar[i];
        vector_sum += vector[i];
    }
    const double scalar_mean = scalar_sum / (double)SAMPLES;
    const double vector_mean = vector_sum / (double)SAMPLES;

    printf(
        "QWEN38_BONSAI2_PTQ1_QH_REAL_GEOMETRY_PASS "
        "label=%s rows=%zu n=%zu "
        "scalar_mean_seconds=%.9f "
        "vector_mean_seconds=%.9f speedup=%.4fx "
        "scalar_samples=[%.9f,%.9f,%.9f,%.9f] "
        "vector_samples=[%.9f,%.9f,%.9f,%.9f]\n",
        label, rows, n,
        scalar_mean, vector_mean,
        vector_mean > 0.0 ? scalar_mean / vector_mean : 0.0,
        scalar[0], scalar[1], scalar[2], scalar[3],
        vector[0], vector[1], vector[2], vector[3]);

    free(out);
    free(activation);
    free(weights);
    return 0;
}


int main(void) {
    if (check_case(257, 128) != 0) return 1;
    if (check_case(257, 5120) != 0) return 2;
    if (check_case(17, 17408) != 0) return 3;
    if (check_qh_vector_case(257, 5120) != 0) return 4;
    if (check_qh_vector_case(17, 17408) != 0) return 5;
    if (benchmark() != 0) return 6;
    if (benchmark_qh_geometry(17408, 5120, "ffn-gate-up") != 0) return 7;
    if (benchmark_qh_geometry(5120, 17408, "ffn-down") != 0) return 8;
    puts("QWEN38_BONSAI2_PTQ1_CACHED_SCALES_BITWISE_PASS");
    return 0;
}
