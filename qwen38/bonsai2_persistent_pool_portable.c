/* Portable persistent row pool for Bonsai 2 PTQ1_0/PQ2_0 matvec.
 *
 * The arithmetic for each output row is delegated unchanged to the proven
 * native Bonsai 2 kernel. Threads own disjoint row ranges, so output ordering
 * and per-row floating-point accumulation semantics are preserved exactly.
 */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "qwen_thread_compat.h"
#include "bonsai2_quant_dot_avx2.c"

typedef enum {
    QWEN_BONSAI2_KIND_PTQ1 = 1,
    QWEN_BONSAI2_KIND_PQ2 = 2,
} qwen_bonsai2_kind;

typedef struct qwen_bonsai2_pool qwen_bonsai2_pool;

typedef struct {
    qwen_bonsai2_pool *pool;
    qwen_thread_t thread;
    int ith;
    int rc;
} qwen_bonsai2_worker;

struct qwen_bonsai2_pool {
    qwen_mutex_t mutex;
    qwen_cond_t work_cond;
    qwen_cond_t done_cond;
    uint64_t generation;
    int done_workers;
    int stop;

    int n_threads;
    size_t max_rows;
    qwen_bonsai2_worker *workers;

    qwen_bonsai2_kind kind;
    const uint8_t *weights;
    size_t weights_bytes;
    size_t rows;
    size_t n;
    const uint8_t *activation;
    size_t activation_bytes;
    float *out;

    size_t scratch_floats;
    float *scratch0;
    float *scratch1;
    size_t scratch_q8_bytes;
    uint8_t *scratch_q8;

    uint64_t calls;
};

static size_t qwen_bonsai2_pool_row_bytes(qwen_bonsai2_kind kind, size_t n) {
    if (n == 0 || n % 128 != 0) return 0;
    if (kind == QWEN_BONSAI2_KIND_PTQ1) {
        return (n / 128) * QWEN_BLOCK_PTQ1_0;
    }
    if (kind == QWEN_BONSAI2_KIND_PQ2) {
        return (n / 128) * QWEN_BLOCK_PQ2_0;
    }
    return 0;
}

static int qwen_bonsai2_pool_compute(qwen_bonsai2_worker *w) {
    qwen_bonsai2_pool *p = w->pool;
    const size_t begin = p->rows * (size_t)w->ith / (size_t)p->n_threads;
    const size_t end = p->rows * (size_t)(w->ith + 1) / (size_t)p->n_threads;
    const size_t local_rows = end - begin;
    if (local_rows == 0) return 0;

    const size_t row_bytes = qwen_bonsai2_pool_row_bytes(p->kind, p->n);
    if (row_bytes == 0) return -20;
    const uint8_t *local_weights = p->weights + begin * row_bytes;
    const size_t local_bytes = local_rows * row_bytes;
    float *local_out = p->out + begin;

    if (p->kind == QWEN_BONSAI2_KIND_PTQ1) {
        return qwen_bonsai2_matvec_ptq1_0_q8_0(
            local_weights, local_bytes, local_rows, p->n,
            p->activation, p->activation_bytes, local_out);
    }
    if (p->kind == QWEN_BONSAI2_KIND_PQ2) {
        return qwen_bonsai2_matvec_pq2_0_q8_0(
            local_weights, local_bytes, local_rows, p->n,
            p->activation, p->activation_bytes, local_out);
    }
    return -21;
}

static QWEN_THREAD_RET qwen_bonsai2_worker_main(void *opaque) {
    qwen_bonsai2_worker *w = (qwen_bonsai2_worker *)opaque;
    qwen_bonsai2_pool *p = w->pool;
    uint64_t seen = 0;

    qwen_mutex_lock(&p->mutex);
    for (;;) {
        while (!p->stop && p->generation == seen) {
            qwen_cond_wait(&p->work_cond, &p->mutex);
        }
        if (p->stop) {
            qwen_mutex_unlock(&p->mutex);
            QWEN_THREAD_RETURN;
        }
        seen = p->generation;
        qwen_mutex_unlock(&p->mutex);

        w->rc = qwen_bonsai2_pool_compute(w);

        qwen_mutex_lock(&p->mutex);
        p->done_workers += 1;
        if (p->done_workers == p->n_threads - 1) {
            qwen_cond_signal(&p->done_cond);
        }
    }
}

QWEN_EXPORT void *qwen_bonsai2_pool_create(int n_threads, size_t max_rows) {
    if (n_threads < 1 || n_threads > 64 || max_rows == 0) return NULL;

    qwen_bonsai2_pool *p = (qwen_bonsai2_pool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_threads = n_threads;
    p->max_rows = max_rows;
    p->scratch_floats = max_rows;
    p->scratch_q8_bytes =
        ((max_rows + 31u) / 32u) * QWEN_BLOCK_Q8_0;
    p->scratch0 = (float *)malloc(max_rows * sizeof(float));
    p->scratch1 = (float *)malloc(max_rows * sizeof(float));
    p->scratch_q8 = (uint8_t *)malloc(p->scratch_q8_bytes);
    if (!p->scratch0 || !p->scratch1 || !p->scratch_q8) {
        free(p->scratch_q8);
        free(p->scratch1);
        free(p->scratch0);
        free(p);
        return NULL;
    }

    if (qwen_mutex_init(&p->mutex) != 0) {
        free(p->scratch_q8);
        free(p->scratch1);
        free(p->scratch0);
        free(p); return NULL;
    }
    if (qwen_cond_init(&p->work_cond) != 0) {
        qwen_mutex_destroy(&p->mutex);
        free(p->scratch_q8);
        free(p->scratch1);
        free(p->scratch0);
        free(p); return NULL;
    }
    if (qwen_cond_init(&p->done_cond) != 0) {
        qwen_cond_destroy(&p->work_cond);
        qwen_mutex_destroy(&p->mutex);
        free(p); return NULL;
    }

    p->workers = (qwen_bonsai2_worker *)calloc(
        (size_t)n_threads, sizeof(*p->workers));
    if (!p->workers) {
        qwen_cond_destroy(&p->done_cond);
        qwen_cond_destroy(&p->work_cond);
        qwen_mutex_destroy(&p->mutex);
        free(p->scratch_q8);
        free(p->scratch1);
        free(p->scratch0);
        free(p);
        return NULL;
    }

    for (int i = 0; i < n_threads; ++i) {
        p->workers[i].pool = p;
        p->workers[i].ith = i;
        if (i > 0 && qwen_thread_create(
                &p->workers[i].thread,
                qwen_bonsai2_worker_main, &p->workers[i]) != 0) {
            qwen_mutex_lock(&p->mutex);
            p->stop = 1;
            qwen_cond_broadcast(&p->work_cond);
            qwen_mutex_unlock(&p->mutex);
            for (int j = 1; j < i; ++j) {
                (void)qwen_thread_join(p->workers[j].thread);
            }
            free(p->workers);
            qwen_cond_destroy(&p->done_cond);
            qwen_cond_destroy(&p->work_cond);
            qwen_mutex_destroy(&p->mutex);
            free(p->scratch_q8);
            free(p->scratch1);
            free(p->scratch0);
            free(p);
            return NULL;
        }
    }
    return p;
}

QWEN_EXPORT void qwen_bonsai2_pool_destroy(void *opaque) {
    qwen_bonsai2_pool *p = (qwen_bonsai2_pool *)opaque;
    if (!p) return;
    if (p->n_threads > 1) {
        qwen_mutex_lock(&p->mutex);
        p->stop = 1;
        qwen_cond_broadcast(&p->work_cond);
        qwen_mutex_unlock(&p->mutex);
        for (int i = 1; i < p->n_threads; ++i) {
            (void)qwen_thread_join(p->workers[i].thread);
        }
    }
    free(p->workers);
    free(p->scratch_q8);
    free(p->scratch1);
    free(p->scratch0);
    qwen_cond_destroy(&p->done_cond);
    qwen_cond_destroy(&p->work_cond);
    qwen_mutex_destroy(&p->mutex);
    free(p);
}

static int qwen_bonsai2_pool_matvec(
        void *opaque, qwen_bonsai2_kind kind,
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    qwen_bonsai2_pool *p = (qwen_bonsai2_pool *)opaque;
    if (!p || !weights || !activation || !out || rows == 0 || n == 0) return -1;
    if (rows > p->max_rows) return -2;
    const size_t row_bytes = qwen_bonsai2_pool_row_bytes(kind, n);
    if (row_bytes == 0) return -3;
    if (weights_bytes != rows * row_bytes) return -4;
    if (activation_bytes != (n / 32) * QWEN_BLOCK_Q8_0) return -5;

    if (p->n_threads == 1) {
        int rc = kind == QWEN_BONSAI2_KIND_PTQ1
            ? qwen_bonsai2_matvec_ptq1_0_q8_0(
                weights, weights_bytes, rows, n,
                activation, activation_bytes, out)
            : qwen_bonsai2_matvec_pq2_0_q8_0(
                weights, weights_bytes, rows, n,
                activation, activation_bytes, out);
        if (rc == 0) p->calls += 1;
        return rc;
    }

    qwen_mutex_lock(&p->mutex);
    p->kind = kind;
    p->weights = weights;
    p->weights_bytes = weights_bytes;
    p->rows = rows;
    p->n = n;
    p->activation = activation;
    p->activation_bytes = activation_bytes;
    p->out = out;
    p->done_workers = 0;
    for (int i = 0; i < p->n_threads; ++i) p->workers[i].rc = 0;
    p->generation += 1;
    qwen_cond_broadcast(&p->work_cond);
    qwen_mutex_unlock(&p->mutex);

    p->workers[0].rc = qwen_bonsai2_pool_compute(&p->workers[0]);

    qwen_mutex_lock(&p->mutex);
    while (p->done_workers != p->n_threads - 1) {
        qwen_cond_wait(&p->done_cond, &p->mutex);
    }
    qwen_mutex_unlock(&p->mutex);

    int rc = p->workers[0].rc;
    for (int i = 1; i < p->n_threads && rc == 0; ++i) {
        if (p->workers[i].rc != 0) rc = p->workers[i].rc;
    }
    if (rc == 0) p->calls += 1;
    return rc;
}

QWEN_EXPORT int qwen_bonsai2_pool_matvec_ptq1_0(
        void *opaque,
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    return qwen_bonsai2_pool_matvec(
        opaque, QWEN_BONSAI2_KIND_PTQ1,
        weights, weights_bytes, rows, n,
        activation, activation_bytes, out);
}

QWEN_EXPORT int qwen_bonsai2_pool_matvec_pq2_0(
        void *opaque,
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activation, size_t activation_bytes, float *out) {
    return qwen_bonsai2_pool_matvec(
        opaque, QWEN_BONSAI2_KIND_PQ2,
        weights, weights_bytes, rows, n,
        activation, activation_bytes, out);
}

QWEN_EXPORT int qwen_bonsai2_pool_ffn_ptq1_0(
        void *opaque,
        const float *x,
        size_t hidden,
        size_t intermediate,
        const uint8_t *gate_weights,
        size_t gate_bytes,
        const uint8_t *up_weights,
        size_t up_bytes,
        const uint8_t *down_weights,
        size_t down_bytes,
        size_t block_size,
        const int8_t *sign_hidden,
        const int8_t *sign_intermediate,
        float *out) {
    qwen_bonsai2_pool *p = (qwen_bonsai2_pool *)opaque;
    if (!p || !x || !gate_weights || !up_weights || !down_weights || !out) {
        return -1;
    }
    if (hidden == 0 || intermediate == 0 ||
        hidden > p->scratch_floats || intermediate > p->scratch_floats ||
        hidden % 128 != 0 || intermediate % 128 != 0 ||
        block_size == 0) {
        return -2;
    }

    const size_t gate_row = qwen_bonsai2_pool_row_bytes(
        QWEN_BONSAI2_KIND_PTQ1, hidden);
    const size_t down_row = qwen_bonsai2_pool_row_bytes(
        QWEN_BONSAI2_KIND_PTQ1, intermediate);
    if (gate_row == 0 || down_row == 0 ||
        gate_bytes != intermediate * gate_row ||
        up_bytes != intermediate * gate_row ||
        down_bytes != hidden * down_row) {
        return -3;
    }

    const size_t hidden_q8 = (hidden / 32u) * QWEN_BLOCK_Q8_0;
    const size_t intermediate_q8 =
        (intermediate / 32u) * QWEN_BLOCK_Q8_0;
    if (hidden_q8 > p->scratch_q8_bytes ||
        intermediate_q8 > p->scratch_q8_bytes) {
        return -4;
    }

    memcpy(p->scratch0, x, hidden * sizeof(float));
    int rc = qwen_bonsai2_fwht_blocks(
        p->scratch0, hidden, block_size, sign_hidden);
    if (rc != 0) return -10 + rc;
    rc = qwen_quantize_q8_0_scalar(
        p->scratch0, hidden, p->scratch_q8, hidden_q8);
    if (rc != 0) return -20 + rc;

    rc = qwen_bonsai2_pool_matvec(
        p, QWEN_BONSAI2_KIND_PTQ1,
        gate_weights, gate_bytes, intermediate, hidden,
        p->scratch_q8, hidden_q8, p->scratch0);
    if (rc != 0) return -30 + rc;
    rc = qwen_bonsai2_pool_matvec(
        p, QWEN_BONSAI2_KIND_PTQ1,
        up_weights, up_bytes, intermediate, hidden,
        p->scratch_q8, hidden_q8, p->scratch1);
    if (rc != 0) return -40 + rc;

    rc = qwen_bonsai2_swiglu_f32(
        p->scratch0, p->scratch1, intermediate, p->scratch0);
    if (rc != 0) return -50 + rc;

    rc = qwen_bonsai2_fwht_blocks(
        p->scratch0, intermediate, block_size, sign_intermediate);
    if (rc != 0) return -60 + rc;
    rc = qwen_quantize_q8_0_scalar(
        p->scratch0, intermediate,
        p->scratch_q8, intermediate_q8);
    if (rc != 0) return -70 + rc;

    rc = qwen_bonsai2_pool_matvec(
        p, QWEN_BONSAI2_KIND_PTQ1,
        down_weights, down_bytes, hidden, intermediate,
        p->scratch_q8, intermediate_q8, out);
    if (rc != 0) return -80 + rc;
    return 0;
}

QWEN_EXPORT int qwen_bonsai2_pool_threads(void *opaque) {
    const qwen_bonsai2_pool *p = (const qwen_bonsai2_pool *)opaque;
    return p ? p->n_threads : 0;
}

QWEN_EXPORT uint64_t qwen_bonsai2_pool_calls(void *opaque) {
    const qwen_bonsai2_pool *p = (const qwen_bonsai2_pool *)opaque;
    return p ? p->calls : 0;
}


#ifdef QWEN_BONSAI2_PORTABLE_POOL_SELFTEST
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { N = 128, ROWS = 257 };
    uint8_t *weights = (uint8_t *)calloc((size_t)ROWS, QWEN_BLOCK_PTQ1_0);
    uint8_t activation[4 * QWEN_BLOCK_Q8_0] = {0};
    float *reference = (float *)calloc((size_t)ROWS, sizeof(float));
    float *candidate = (float *)calloc((size_t)ROWS, sizeof(float));
    if (!weights || !reference || !candidate) return 20;

    for (int r = 0; r < ROWS; ++r) {
        uint8_t *block = weights + (size_t)r * QWEN_BLOCK_PTQ1_0;
        block[26] = 0x00;
        block[27] = 0x3c; /* fp16 1.0; zero trits decode to -1 */
    }
    for (int k = 0; k < 4; ++k) {
        uint8_t *block = activation + k * QWEN_BLOCK_Q8_0;
        block[0] = 0x00;
        block[1] = 0x3c; /* fp16 1.0 */
        memset(block + 2, 1, 32);
    }

    if (qwen_bonsai2_matvec_ptq1_0_q8_0(
            weights, (size_t)ROWS * QWEN_BLOCK_PTQ1_0,
            ROWS, N, activation, sizeof(activation), reference) != 0) return 21;

    const int threads_to_test[] = {1, 2, 4};
    for (int it = 0; it < 3; ++it) {
        const int nth = threads_to_test[it];
        void *pool = qwen_bonsai2_pool_create(nth, ROWS);
        if (!pool) return 30 + it;
        memset(candidate, 0, (size_t)ROWS * sizeof(float));
        const int rc = qwen_bonsai2_pool_matvec_ptq1_0(
            pool,
            weights, (size_t)ROWS * QWEN_BLOCK_PTQ1_0,
            ROWS, N, activation, sizeof(activation), candidate);
        if (rc != 0) {
            qwen_bonsai2_pool_destroy(pool);
            return 40 + it;
        }
        if (memcmp(reference, candidate, (size_t)ROWS * sizeof(float)) != 0) {
            qwen_bonsai2_pool_destroy(pool);
            return 50 + it;
        }
        if (qwen_bonsai2_pool_threads(pool) != nth ||
            qwen_bonsai2_pool_calls(pool) != 1) {
            qwen_bonsai2_pool_destroy(pool);
            return 60 + it;
        }
        qwen_bonsai2_pool_destroy(pool);
    }

    free(weights);
    free(reference);
    free(candidate);
    puts("QWEN38_BONSAI2_PORTABLE_POOL_PASS");
    return 0;
}
#endif
