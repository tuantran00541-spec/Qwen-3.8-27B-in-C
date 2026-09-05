/* Exact row-parallel overlay for Qwen3.8 CPU research.
 *
 * Each output row is mathematically independent.  Threads own disjoint row
 * ranges and call the already-proven exact AVX2 vec-dot kernels, so the
 * floating-point accumulation order inside every output element is unchanged.
 * This file is a Linux/pthread proof gate only; a promoted Windows runtime
 * must use the native Windows thread backend rather than depending on pthreads.
 */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gguf_quant_dot_avx2.c"

typedef enum {
    QWEN_ROW_Q6_K_Q8_K = 0,
    QWEN_ROW_Q8_0_Q8_0 = 1,
} qwen_row_kind;

typedef struct {
    qwen_row_kind kind;
    const uint8_t *weights;
    size_t row_bytes;
    size_t n;
    const uint8_t *activation;
    size_t activation_bytes;
    float *out;
    size_t row_begin;
    size_t row_end;
    int rc;
} qwen_row_task;

static void *qwen_row_worker(void *opaque) {
    qwen_row_task *t = (qwen_row_task *)opaque;
    t->rc = 0;
    for (size_t r = t->row_begin; r < t->row_end; ++r) {
        const uint8_t *w = t->weights + r * t->row_bytes;
        float v;
        if (t->kind == QWEN_ROW_Q6_K_Q8_K) {
            v = qwen_vec_dot_q6_k_q8_k_scalar(
                w, t->row_bytes, t->activation, t->activation_bytes, t->n);
        } else {
            v = qwen_vec_dot_q8_0_q8_0_scalar(
                w, t->row_bytes, t->activation, t->activation_bytes, t->n);
        }
        if (!isfinite(v)) {
            t->rc = -10;
            return NULL;
        }
        t->out[r] = v;
    }
    return NULL;
}

static int qwen_matvec_rows_parallel_exact(
        qwen_row_kind kind,
        const uint8_t *weights,
        size_t weights_bytes,
        size_t rows,
        size_t n,
        const uint8_t *activation,
        size_t activation_bytes,
        float *out,
        int n_threads) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n_threads < 1) return -1;
    size_t row_bytes;
    if (kind == QWEN_ROW_Q6_K_Q8_K) {
        if (n % QWEN_QK_K != 0) return -2;
        row_bytes = (n / QWEN_QK_K) * QWEN_BLOCK_Q6_K;
        if (activation_bytes != (n / QWEN_QK_K) * QWEN_BLOCK_Q8_K) return -3;
    } else {
        if (n % QWEN_QK8_0 != 0) return -4;
        row_bytes = (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
        if (activation_bytes != row_bytes) return -5;
    }
    if (weights_bytes != rows * row_bytes) return -6;

    if ((size_t)n_threads > rows) n_threads = (int)rows;
    if (n_threads == 1) {
        qwen_row_task t = {
            .kind = kind, .weights = weights, .row_bytes = row_bytes, .n = n,
            .activation = activation, .activation_bytes = activation_bytes,
            .out = out, .row_begin = 0, .row_end = rows, .rc = 0,
        };
        qwen_row_worker(&t);
        return t.rc;
    }

    pthread_t *threads = (pthread_t *)calloc((size_t)n_threads, sizeof(*threads));
    qwen_row_task *tasks = (qwen_row_task *)calloc((size_t)n_threads, sizeof(*tasks));
    if (!threads || !tasks) {
        free(threads); free(tasks); return -7;
    }

    int created = 0;
    for (int i = 0; i < n_threads; ++i) {
        const size_t begin = (rows * (size_t)i) / (size_t)n_threads;
        const size_t end = (rows * (size_t)(i + 1)) / (size_t)n_threads;
        tasks[i] = (qwen_row_task){
            .kind = kind, .weights = weights, .row_bytes = row_bytes, .n = n,
            .activation = activation, .activation_bytes = activation_bytes,
            .out = out, .row_begin = begin, .row_end = end, .rc = 0,
        };
        if (pthread_create(&threads[i], NULL, qwen_row_worker, &tasks[i]) != 0) {
            for (int j = 0; j < created; ++j) pthread_join(threads[j], NULL);
            free(threads); free(tasks); return -8;
        }
        ++created;
    }
    int rc = 0;
    for (int i = 0; i < created; ++i) {
        if (pthread_join(threads[i], NULL) != 0 && rc == 0) rc = -9;
        if (tasks[i].rc != 0 && rc == 0) rc = tasks[i].rc;
    }
    free(threads); free(tasks);
    return rc;
}

#ifdef QWEN_ROW_PARALLEL_SELFTEST
static uint32_t rng_state = 0x9f2d471bu;
static uint32_t rng_u32(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f32(void) { return ((int32_t)(rng_u32() >> 8) % 20001 - 10000) / 1009.0f; }

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
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int run_q6(int n_threads, double *elapsed) {
    enum { N = 5120, ROWS = 4096 };
    const size_t wr = (N / QWEN_QK_K) * QWEN_BLOCK_Q6_K;
    const size_t ar = (N / QWEN_QK_K) * QWEN_BLOCK_Q8_K;
    uint8_t *w = (uint8_t *)malloc(ROWS * wr);
    uint8_t *a = (uint8_t *)malloc(ar);
    float *x = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(ROWS * sizeof(float));
    float *cand = (float *)malloc(ROWS * sizeof(float));
    if (!w || !a || !x || !ref || !cand) return 20;
    for (size_t r = 0; r < ROWS; ++r)
        for (size_t ib = 0; ib < N / QWEN_QK_K; ++ib)
            pack_q6_fixture(w + r * wr + ib * QWEN_BLOCK_Q6_K);
    for (int i = 0; i < N; ++i) x[i] = rng_f32();
    if (qwen_quantize_q8_k_scalar(x, N, a, ar) != 0) return 21;
    if (qwen_matvec_q6_k_q8_k_scalar(w, ROWS*wr, ROWS, N, a, ar, ref) != 0) return 22;
    const double t0 = now_s();
    if (qwen_matvec_rows_parallel_exact(QWEN_ROW_Q6_K_Q8_K, w, ROWS*wr, ROWS, N, a, ar, cand, n_threads) != 0) return 23;
    *elapsed = now_s() - t0;
    const int ok = memcmp(ref, cand, ROWS*sizeof(float)) == 0;
    free(w); free(a); free(x); free(ref); free(cand);
    return ok ? 0 : 24;
}

static int run_q8(int n_threads, double *elapsed) {
    enum { N = 5120, ROWS = 4096 };
    const size_t rb = (N / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;
    uint8_t *w = (uint8_t *)malloc(ROWS * rb);
    uint8_t *a = (uint8_t *)malloc(rb);
    float *x = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(ROWS * sizeof(float));
    float *cand = (float *)malloc(ROWS * sizeof(float));
    if (!w || !a || !x || !ref || !cand) return 30;
    for (size_t r = 0; r < ROWS; ++r) {
        for (int i = 0; i < N; ++i) x[i] = rng_f32();
        if (qwen_quantize_q8_0_scalar(x, N, w+r*rb, rb) != 0) return 31;
    }
    for (int i = 0; i < N; ++i) x[i] = rng_f32();
    if (qwen_quantize_q8_0_scalar(x, N, a, rb) != 0) return 32;
    if (qwen_matvec_q8_0_q8_0_scalar(w, ROWS*rb, ROWS, N, a, rb, ref) != 0) return 33;
    const double t0 = now_s();
    if (qwen_matvec_rows_parallel_exact(QWEN_ROW_Q8_0_Q8_0, w, ROWS*rb, ROWS, N, a, rb, cand, n_threads) != 0) return 34;
    *elapsed = now_s() - t0;
    const int ok = memcmp(ref, cand, ROWS*sizeof(float)) == 0;
    free(w); free(a); free(x); free(ref); free(cand);
    return ok ? 0 : 35;
}

int main(void) {
    const int thread_counts[] = {1, 2, 4};
    double q6[3] = {0}, q8[3] = {0};
    for (int i = 0; i < 3; ++i) {
        int rc = run_q6(thread_counts[i], &q6[i]);
        if (rc != 0) { fprintf(stderr, "Q6 rows exact failed threads=%d rc=%d\n", thread_counts[i], rc); return rc; }
        rc = run_q8(thread_counts[i], &q8[i]);
        if (rc != 0) { fprintf(stderr, "Q8 rows exact failed threads=%d rc=%d\n", thread_counts[i], rc); return rc; }
    }
    printf("Q6 row-parallel t1=%.6f t2=%.6f t4=%.6f speedup4=%.4fx\n", q6[0], q6[1], q6[2], q6[0]/q6[2]);
    printf("Q8 row-parallel t1=%.6f t2=%.6f t4=%.6f speedup4=%.4fx\n", q8[0], q8[1], q8[2], q8[0]/q8[2]);
    puts("QWEN38_ROW_PARALLEL_BITWISE_PASS");
    return 0;
}
#endif
