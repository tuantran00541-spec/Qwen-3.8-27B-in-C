#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gguf_quant_dot_avx2.c"

static uint32_t bench_rng(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static void put_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)(v >> 8);
}

static double now_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(90);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(const double *x, size_t n) {
    double *tmp = (double *)malloc(n * sizeof(*tmp));
    if (!tmp) exit(91);
    memcpy(tmp, x, n * sizeof(*tmp));
    qsort(tmp, n, sizeof(*tmp), cmp_double);
    const double out = (n & 1u) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
    return out;
}

static void fill_q6_matrix(uint8_t *w, size_t rows, size_t n, uint32_t *s) {
    const size_t blocks_per_row = n / QWEN_QK_K;
    const uint16_t d = qwen_f32_to_f16(0.03125f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < blocks_per_row; ++b) {
            uint8_t *blk = w + (r * blocks_per_row + b) * QWEN_BLOCK_Q6_K;
            for (int i = 0; i < 128; ++i) blk[i] = (uint8_t)(bench_rng(s) >> 24);
            for (int i = 0; i < 64; ++i) blk[128 + i] = (uint8_t)(bench_rng(s) >> 24);
            for (int i = 0; i < 16; ++i) {
                int v = (int)((bench_rng(s) >> 24) % 31u) - 15;
                blk[192 + i] = (uint8_t)(int8_t)v;
            }
            put_u16_le(blk + 208, d);
        }
    }
}

static void fill_q8_0_matrix(uint8_t *w, size_t rows, size_t n, uint32_t *s) {
    const size_t blocks_per_row = n / QWEN_QK8_0;
    const uint16_t d = qwen_f32_to_f16(0.03125f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < blocks_per_row; ++b) {
            uint8_t *blk = w + (r * blocks_per_row + b) * QWEN_BLOCK_Q8_0;
            put_u16_le(blk, d);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(int8_t)(bench_rng(s) >> 24);
        }
    }
}

static void fill_activation(float *x, size_t n, uint32_t *s) {
    for (size_t i = 0; i < n; ++i) {
        const int32_t v = (int32_t)((bench_rng(s) >> 8) % 20001u) - 10000;
        x[i] = (float)v / 997.0f;
    }
}

static void run_q6_case(const char *name, size_t rows, size_t n, int rounds, uint32_t *seed) {
    const size_t wr = (n / QWEN_QK_K) * QWEN_BLOCK_Q6_K;
    const size_t ar = (n / QWEN_QK_K) * QWEN_BLOCK_Q8_K;
    const size_t wb = rows * wr;

    uint8_t *weights = (uint8_t *)malloc(wb);
    uint8_t *activation = (uint8_t *)malloc(ar);
    float *x = (float *)malloc(n * sizeof(*x));
    float *ref = (float *)malloc(rows * sizeof(*ref));
    float *simd = (float *)malloc(rows * sizeof(*simd));
    double *scalar_t = (double *)malloc((size_t)rounds * 2u * sizeof(*scalar_t));
    double *avx2_t = (double *)malloc((size_t)rounds * 2u * sizeof(*avx2_t));
    if (!weights || !activation || !x || !ref || !simd || !scalar_t || !avx2_t) exit(92);

    fill_q6_matrix(weights, rows, n, seed);
    fill_activation(x, n, seed);
    if (qwen_quantize_q8_k_scalar(x, n, activation, ar) != 0) exit(93);

    if (qwen_matvec_q6_k_q8_k_reference(weights, wb, rows, n, activation, ar, ref) != 0) exit(94);
    if (qwen_matvec_q6_k_q8_k_scalar(weights, wb, rows, n, activation, ar, simd) != 0) exit(95);
    if (memcmp(ref, simd, rows * sizeof(*ref)) != 0) {
        fprintf(stderr, "QWEN38_AVX2_BENCH_EXACT_FAIL case=%s\n", name);
        exit(10);
    }

    /* Warm both paths before timing. */
    qwen_matvec_q6_k_q8_k_reference(weights, wb, rows, n, activation, ar, ref);
    qwen_matvec_q6_k_q8_k_scalar(weights, wb, rows, n, activation, ar, simd);

    size_t ns = 0, nv = 0;
    volatile float sink = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        double t0 = now_s();
        qwen_matvec_q6_k_q8_k_reference(weights, wb, rows, n, activation, ar, ref);
        double t1 = now_s();
        scalar_t[ns++] = t1 - t0;
        sink += ref[(size_t)r % rows];

        t0 = now_s();
        qwen_matvec_q6_k_q8_k_scalar(weights, wb, rows, n, activation, ar, simd);
        t1 = now_s();
        avx2_t[nv++] = t1 - t0;
        sink += simd[((size_t)r + 1u) % rows];

        t0 = now_s();
        qwen_matvec_q6_k_q8_k_scalar(weights, wb, rows, n, activation, ar, simd);
        t1 = now_s();
        avx2_t[nv++] = t1 - t0;
        sink += simd[((size_t)r + 2u) % rows];

        t0 = now_s();
        qwen_matvec_q6_k_q8_k_reference(weights, wb, rows, n, activation, ar, ref);
        t1 = now_s();
        scalar_t[ns++] = t1 - t0;
        sink += ref[((size_t)r + 3u) % rows];
    }

    const double ms = median(scalar_t, ns);
    const double ma = median(avx2_t, nv);
    printf("{\"case\":\"%s\",\"quant\":\"Q6_KxQ8_K\",\"rows\":%zu,\"n\":%zu,\"weight_bytes\":%zu,\"samples_each\":%zu,\"scalar_median_s\":%.9f,\"avx2_median_s\":%.9f,\"speedup\":%.6f,\"bitwise_exact\":true,\"sink\":%.9g}\n",
           name, rows, n, wb, ns, ms, ma, ms / ma, (double)sink);

    free(avx2_t); free(scalar_t); free(simd); free(ref); free(x); free(activation); free(weights);
}

static void run_q8_case(const char *name, size_t rows, size_t n, int rounds, uint32_t *seed) {
    const size_t rowb = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    const size_t wb = rows * rowb;
    uint8_t *weights = (uint8_t *)malloc(wb);
    uint8_t *activation = (uint8_t *)malloc(rowb);
    float *x = (float *)malloc(n * sizeof(*x));
    float *ref = (float *)malloc(rows * sizeof(*ref));
    float *simd = (float *)malloc(rows * sizeof(*simd));
    double *scalar_t = (double *)malloc((size_t)rounds * 2u * sizeof(*scalar_t));
    double *avx2_t = (double *)malloc((size_t)rounds * 2u * sizeof(*avx2_t));
    if (!weights || !activation || !x || !ref || !simd || !scalar_t || !avx2_t) exit(96);

    fill_q8_0_matrix(weights, rows, n, seed);
    fill_activation(x, n, seed);
    if (qwen_quantize_q8_0_scalar(x, n, activation, rowb) != 0) exit(97);
    if (qwen_matvec_q8_0_q8_0_reference(weights, wb, rows, n, activation, rowb, ref) != 0) exit(98);
    if (qwen_matvec_q8_0_q8_0_scalar(weights, wb, rows, n, activation, rowb, simd) != 0) exit(99);
    if (memcmp(ref, simd, rows * sizeof(*ref)) != 0) {
        fprintf(stderr, "QWEN38_AVX2_BENCH_EXACT_FAIL case=%s\n", name);
        exit(11);
    }

    qwen_matvec_q8_0_q8_0_reference(weights, wb, rows, n, activation, rowb, ref);
    qwen_matvec_q8_0_q8_0_scalar(weights, wb, rows, n, activation, rowb, simd);

    size_t ns = 0, nv = 0;
    volatile float sink = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        double t0 = now_s();
        qwen_matvec_q8_0_q8_0_reference(weights, wb, rows, n, activation, rowb, ref);
        double t1 = now_s();
        scalar_t[ns++] = t1 - t0;
        sink += ref[(size_t)r % rows];

        t0 = now_s();
        qwen_matvec_q8_0_q8_0_scalar(weights, wb, rows, n, activation, rowb, simd);
        t1 = now_s();
        avx2_t[nv++] = t1 - t0;
        sink += simd[((size_t)r + 1u) % rows];

        t0 = now_s();
        qwen_matvec_q8_0_q8_0_scalar(weights, wb, rows, n, activation, rowb, simd);
        t1 = now_s();
        avx2_t[nv++] = t1 - t0;
        sink += simd[((size_t)r + 2u) % rows];

        t0 = now_s();
        qwen_matvec_q8_0_q8_0_reference(weights, wb, rows, n, activation, rowb, ref);
        t1 = now_s();
        scalar_t[ns++] = t1 - t0;
        sink += ref[((size_t)r + 3u) % rows];
    }

    const double ms = median(scalar_t, ns);
    const double ma = median(avx2_t, nv);
    printf("{\"case\":\"%s\",\"quant\":\"Q8_0xQ8_0\",\"rows\":%zu,\"n\":%zu,\"weight_bytes\":%zu,\"samples_each\":%zu,\"scalar_median_s\":%.9f,\"avx2_median_s\":%.9f,\"speedup\":%.6f,\"bitwise_exact\":true,\"sink\":%.9g}\n",
           name, rows, n, wb, ns, ms, ma, ms / ma, (double)sink);

    free(avx2_t); free(scalar_t); free(simd); free(ref); free(x); free(activation); free(weights);
}

int main(void) {
    uint32_t seed = 0x51a7c0deu;
    puts("QWEN38_AVX2_BENCH_BEGIN");
    /* Actual Qwen3.8 FFN matrix geometries: gate/up and down projections. */
    run_q6_case("ffn_gate_up_17408x5120", 17408, 5120, 6, &seed);
    run_q6_case("ffn_down_5120x17408", 5120, 17408, 6, &seed);
    /* Representative decoder Q8 projection geometry; kernel-only, not LM-head I/O. */
    run_q8_case("q8_projection_6144x5120", 6144, 5120, 6, &seed);
    puts("QWEN38_AVX2_BENCH_EXACT_PASS");
    return 0;
}
