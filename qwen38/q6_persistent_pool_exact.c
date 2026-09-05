/* Exact persistent row-parallel Q6_K matvec-many experiment for Qwen3.8.
 *
 * This is intentionally Q6_K-only.  Q8_0 continues to use the proven noalloc
 * bridge unchanged.  The worker pool partitions only independent output rows.
 * Each row slice calls the already-proven qwen_matvec_many_q6_k_q8_k_exact()
 * implementation, so Q6 unpacking, integer dot products, F32 lane accumulation,
 * and final lane reduction order inside every output element are unchanged.
 *
 * Linux/pthread is used for this experimental performance gate.  This file is
 * not the final promoted Windows backend; a positive result must later move the
 * synchronization layer behind a POSIX/Win32 abstraction before production use.
 */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gguf_quant_matvec_many_noalloc_bridge.c"

typedef struct qwen_q6_pool qwen_q6_pool;

typedef struct {
    qwen_q6_pool *pool;
    pthread_t thread;
    int ith;
    int rc;
    float *scratch;
    size_t scratch_floats;
} qwen_q6_worker;

struct qwen_q6_pool {
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    uint64_t generation;
    int done_workers;
    int stop;

    int n_threads;
    size_t max_rows;
    size_t max_vec;
    qwen_q6_worker *workers;

    const uint8_t *weights;
    size_t weights_bytes;
    size_t rows;
    size_t n;
    const uint8_t *activations;
    size_t activation_bytes_each;
    size_t n_vec;
    float *out;

    uint64_t calls;
};

static int qwen_q6_pool_compute_worker(qwen_q6_worker *w) {
    qwen_q6_pool *p = w->pool;
    const size_t rows = p->rows;
    const size_t begin = rows * (size_t)w->ith / (size_t)p->n_threads;
    const size_t end = rows * (size_t)(w->ith + 1) / (size_t)p->n_threads;
    const size_t local_rows = end - begin;
    if (local_rows == 0) {
        return 0;
    }
    const size_t needed = local_rows * p->n_vec;
    if (needed > w->scratch_floats) {
        return -20;
    }
    if (p->n % QWEN_QK_K != 0) {
        return -21;
    }
    const size_t row_bytes = (p->n / QWEN_QK_K) * QWEN_BLOCK_Q6_K;
    const uint8_t *local_weights = p->weights + begin * row_bytes;
    const size_t local_weight_bytes = local_rows * row_bytes;

    const int rc = qwen_matvec_many_q6_k_q8_k_exact(
        local_weights,
        local_weight_bytes,
        local_rows,
        p->n,
        p->activations,
        p->activation_bytes_each,
        p->n_vec,
        w->scratch);
    if (rc != 0) {
        return rc;
    }

    /* The reference many-kernel stores [vector][row].  Each worker owns one
       disjoint row interval and scatters only that interval into the global
       output.  No worker writes another worker's output elements. */
    for (size_t v = 0; v < p->n_vec; ++v) {
        memcpy(
            p->out + v * rows + begin,
            w->scratch + v * local_rows,
            local_rows * sizeof(float));
    }
    return 0;
}

static void *qwen_q6_pool_worker_main(void *opaque) {
    qwen_q6_worker *w = (qwen_q6_worker *)opaque;
    qwen_q6_pool *p = w->pool;
    uint64_t seen_generation = 0;

    pthread_mutex_lock(&p->mutex);
    for (;;) {
        while (!p->stop && p->generation == seen_generation) {
            pthread_cond_wait(&p->work_cond, &p->mutex);
        }
        if (p->stop) {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
        seen_generation = p->generation;
        pthread_mutex_unlock(&p->mutex);

        w->rc = qwen_q6_pool_compute_worker(w);

        pthread_mutex_lock(&p->mutex);
        p->done_workers += 1;
        if (p->done_workers == p->n_threads - 1) {
            pthread_cond_signal(&p->done_cond);
        }
    }
}

void *qwen_q6_pool_create(int n_threads, size_t max_rows, size_t max_vec) {
    if (n_threads < 1 || n_threads > 64 || max_rows == 0 || max_vec == 0) {
        return NULL;
    }

    qwen_q6_pool *p = (qwen_q6_pool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_threads = n_threads;
    p->max_rows = max_rows;
    p->max_vec = max_vec;

    if (pthread_mutex_init(&p->mutex, NULL) != 0) {
        free(p); return NULL;
    }
    if (pthread_cond_init(&p->work_cond, NULL) != 0) {
        pthread_mutex_destroy(&p->mutex); free(p); return NULL;
    }
    if (pthread_cond_init(&p->done_cond, NULL) != 0) {
        pthread_cond_destroy(&p->work_cond);
        pthread_mutex_destroy(&p->mutex);
        free(p); return NULL;
    }

    p->workers = (qwen_q6_worker *)calloc((size_t)n_threads, sizeof(*p->workers));
    if (!p->workers) {
        pthread_cond_destroy(&p->done_cond);
        pthread_cond_destroy(&p->work_cond);
        pthread_mutex_destroy(&p->mutex);
        free(p); return NULL;
    }

    const size_t max_local_rows = (max_rows + (size_t)n_threads - 1) / (size_t)n_threads;
    const size_t scratch_floats = max_local_rows * max_vec;
    for (int i = 0; i < n_threads; ++i) {
        p->workers[i].pool = p;
        p->workers[i].ith = i;
        p->workers[i].scratch_floats = scratch_floats;
        p->workers[i].scratch = (float *)malloc(scratch_floats * sizeof(float));
        if (!p->workers[i].scratch) {
            p->stop = 1;
            pthread_cond_broadcast(&p->work_cond);
            for (int j = 1; j < i; ++j) pthread_join(p->workers[j].thread, NULL);
            for (int j = 0; j <= i; ++j) free(p->workers[j].scratch);
            free(p->workers);
            pthread_cond_destroy(&p->done_cond);
            pthread_cond_destroy(&p->work_cond);
            pthread_mutex_destroy(&p->mutex);
            free(p);
            return NULL;
        }
        if (i > 0 && pthread_create(
                &p->workers[i].thread, NULL,
                qwen_q6_pool_worker_main, &p->workers[i]) != 0) {
            pthread_mutex_lock(&p->mutex);
            p->stop = 1;
            pthread_cond_broadcast(&p->work_cond);
            pthread_mutex_unlock(&p->mutex);
            for (int j = 1; j < i; ++j) pthread_join(p->workers[j].thread, NULL);
            for (int j = 0; j <= i; ++j) free(p->workers[j].scratch);
            free(p->workers);
            pthread_cond_destroy(&p->done_cond);
            pthread_cond_destroy(&p->work_cond);
            pthread_mutex_destroy(&p->mutex);
            free(p);
            return NULL;
        }
    }
    return p;
}

void qwen_q6_pool_destroy(void *opaque) {
    qwen_q6_pool *p = (qwen_q6_pool *)opaque;
    if (!p) return;
    if (p->n_threads > 1) {
        pthread_mutex_lock(&p->mutex);
        p->stop = 1;
        pthread_cond_broadcast(&p->work_cond);
        pthread_mutex_unlock(&p->mutex);
        for (int i = 1; i < p->n_threads; ++i) {
            pthread_join(p->workers[i].thread, NULL);
        }
    }
    for (int i = 0; i < p->n_threads; ++i) free(p->workers[i].scratch);
    free(p->workers);
    pthread_cond_destroy(&p->done_cond);
    pthread_cond_destroy(&p->work_cond);
    pthread_mutex_destroy(&p->mutex);
    free(p);
}

int qwen_q6_pool_matvec_many(
        void *opaque,
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    qwen_q6_pool *p = (qwen_q6_pool *)opaque;
    if (!p || !weights || !activations || !out || rows == 0 || n_vec == 0 || n == 0) return -1;
    if (rows > p->max_rows || n_vec > p->max_vec) return -2;
    if (n % QWEN_QK_K != 0) return -3;
    const size_t nb = n / QWEN_QK_K;
    const size_t wr = nb * QWEN_BLOCK_Q6_K;
    const size_t ar = nb * QWEN_BLOCK_Q8_K;
    if (activation_bytes_each != ar) return -4;
    if (weights_bytes != rows * wr) return -5;

    if (p->n_threads == 1) {
        const int rc = qwen_matvec_many_q6_k_q8_k_exact(
            weights, weights_bytes, rows, n,
            activations, activation_bytes_each, n_vec, out);
        if (rc == 0) p->calls += 1;
        return rc;
    }

    pthread_mutex_lock(&p->mutex);
    p->weights = weights;
    p->weights_bytes = weights_bytes;
    p->rows = rows;
    p->n = n;
    p->activations = activations;
    p->activation_bytes_each = activation_bytes_each;
    p->n_vec = n_vec;
    p->out = out;
    p->done_workers = 0;
    for (int i = 0; i < p->n_threads; ++i) p->workers[i].rc = 0;
    p->generation += 1;
    pthread_cond_broadcast(&p->work_cond);
    pthread_mutex_unlock(&p->mutex);

    p->workers[0].rc = qwen_q6_pool_compute_worker(&p->workers[0]);

    pthread_mutex_lock(&p->mutex);
    while (p->done_workers != p->n_threads - 1) {
        pthread_cond_wait(&p->done_cond, &p->mutex);
    }
    pthread_mutex_unlock(&p->mutex);

    int rc = p->workers[0].rc;
    for (int i = 1; i < p->n_threads && rc == 0; ++i) {
        if (p->workers[i].rc != 0) rc = p->workers[i].rc;
    }
    if (rc == 0) p->calls += 1;
    return rc;
}

uint64_t qwen_q6_pool_calls(void *opaque) {
    const qwen_q6_pool *p = (const qwen_q6_pool *)opaque;
    return p ? p->calls : 0;
}

int qwen_q6_pool_threads(void *opaque) {
    const qwen_q6_pool *p = (const qwen_q6_pool *)opaque;
    return p ? p->n_threads : 0;
}

#ifdef QWEN_Q6_PERSISTENT_POOL_SELFTEST
static uint32_t qwen_pool_rng = 0x6a09e667u;
static uint32_t qwen_pool_u32(void) {
    qwen_pool_rng = qwen_pool_rng * 1664525u + 1013904223u;
    return qwen_pool_rng;
}
static float qwen_pool_f32(void) {
    return ((int32_t)(qwen_pool_u32() >> 8) % 20001 - 10000) / 997.0f;
}

static void qwen_pool_pack_q6(uint8_t out[QWEN_BLOCK_Q6_K]) {
    int8_t scales[16], quants[256];
    for (int i = 0; i < 16; ++i) scales[i] = (int8_t)((qwen_pool_u32() >> 24) % 31 - 15);
    for (int i = 0; i < 256; ++i) quants[i] = (int8_t)((qwen_pool_u32() >> 24) % 64 - 32);
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
            qh[qh_base+l] = (uint8_t)(
                ((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) |
                (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6));
        }
    }
    memcpy(out + 192, scales, 16);
    const uint16_t d = qwen_f32_to_f16(0.03125f);
    out[208] = (uint8_t)(d & 0xffu);
    out[209] = (uint8_t)(d >> 8);
}

static double qwen_pool_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { N = 5120, ROWS = 1024, NV = 11 };
    const size_t wr = (N / QWEN_QK_K) * QWEN_BLOCK_Q6_K;
    const size_t ar = (N / QWEN_QK_K) * QWEN_BLOCK_Q8_K;
    uint8_t *weights = (uint8_t *)malloc((size_t)ROWS * wr);
    uint8_t *acts = (uint8_t *)malloc((size_t)NV * ar);
    float *x = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc((size_t)NV * ROWS * sizeof(float));
    float *cand = (float *)malloc((size_t)NV * ROWS * sizeof(float));
    if (!weights || !acts || !x || !ref || !cand) return 30;

    for (size_t r = 0; r < ROWS; ++r) {
        for (size_t ib = 0; ib < N / QWEN_QK_K; ++ib) {
            qwen_pool_pack_q6(weights + r * wr + ib * QWEN_BLOCK_Q6_K);
        }
    }
    for (int v = 0; v < NV; ++v) {
        for (int i = 0; i < N; ++i) x[i] = qwen_pool_f32();
        if (qwen_quantize_q8_k_scalar(x, N, acts + (size_t)v * ar, ar) != 0) return 31;
    }
    if (qwen_matvec_many_q6_k_q8_k_exact(
            weights, (size_t)ROWS * wr, ROWS, N,
            acts, ar, NV, ref) != 0) return 32;

    const int thread_counts[] = {1, 2, 4};
    for (int it = 0; it < 3; ++it) {
        const int nth = thread_counts[it];
        void *pool = qwen_q6_pool_create(nth, ROWS, NV);
        if (!pool) return 33 + it;
        memset(cand, 0, (size_t)NV * ROWS * sizeof(float));
        const double t0 = qwen_pool_now();
        const int rc = qwen_q6_pool_matvec_many(
            pool, weights, (size_t)ROWS * wr, ROWS, N,
            acts, ar, NV, cand);
        const double sec = qwen_pool_now() - t0;
        if (rc != 0) {
            fprintf(stderr, "persistent Q6 pool rc=%d threads=%d\n", rc, nth);
            qwen_q6_pool_destroy(pool); return 40 + it;
        }
        if (memcmp(ref, cand, (size_t)NV * ROWS * sizeof(float)) != 0) {
            fprintf(stderr, "persistent Q6 pool bitwise mismatch threads=%d\n", nth);
            qwen_q6_pool_destroy(pool); return 50 + it;
        }
        if (qwen_q6_pool_calls(pool) != 1 || qwen_q6_pool_threads(pool) != nth) {
            qwen_q6_pool_destroy(pool); return 60 + it;
        }
        printf("Q6 persistent-pool synthetic threads=%d seconds=%.6f\n", nth, sec);
        qwen_q6_pool_destroy(pool);
    }

    puts("QWEN38_Q6_PERSISTENT_POOL_BITWISE_PASS");
    free(weights); free(acts); free(x); free(ref); free(cand);
    return 0;
}
#endif
