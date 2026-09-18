/* Portable persistent head pool for Qwen3.8/Qwen3.5 Gated-DeltaNet AR state.
 *
 * Token order stays strictly serial. Only the 48 independent value-head state
 * planes are partitioned across persistent workers. Arithmetic inside each head
 * is copied from gdn_state_ar.c so state/output remain bitwise identical to the
 * proven scalar reference regardless of thread count.
 */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "qwen_thread_compat.h"

#define qwen_gdn_ar_step_f32 qwen_gdn_ar_step_f32_reference
#include "gdn_state_ar.c"
#undef qwen_gdn_ar_step_f32

typedef struct qwen_gdn_pool qwen_gdn_pool;

typedef struct {
    qwen_gdn_pool *pool;
    qwen_thread_t thread;
    int ith;
    int rc;
} qwen_gdn_worker;

struct qwen_gdn_pool {
    qwen_mutex_t mutex;
    qwen_cond_t work_cond;
    qwen_cond_t done_cond;
    uint64_t generation;
    int done_workers;
    int stop;

    int n_threads;
    qwen_gdn_worker *workers;

    float *state;
    const float *q;
    const float *k;
    const float *v;
    const float *gate;
    const float *beta;
    float *out;

    uint64_t calls;
};

static int qwen_gdn_compute_worker(qwen_gdn_worker *w) {
    qwen_gdn_pool *p = w->pool;
    const int h_begin = (QWEN_GDN_HEADS * w->ith) / p->n_threads;
    const int h_end = (QWEN_GDN_HEADS * (w->ith + 1)) / p->n_threads;

    for (int h = h_begin; h < h_end; ++h) {
        float *s = p->state + (size_t)h * QWEN_GDN_DIM * QWEN_GDN_DIM;
        const float *qh = p->q + (size_t)h * QWEN_GDN_DIM;
        const float *kh = p->k + (size_t)h * QWEN_GDN_DIM;
        const float *vh = p->v + (size_t)h * QWEN_GDN_DIM;
        float *oh = p->out + (size_t)h * QWEN_GDN_DIM;
        const float decay = expf(p->gate[h]);

        for (int i = 0; i < QWEN_GDN_DIM * QWEN_GDN_DIM; ++i) {
            s[i] *= decay;
        }

        float d[QWEN_GDN_DIM];
        for (int j = 0; j < QWEN_GDN_DIM; ++j) {
            float sk = 0.0f;
            for (int i = 0; i < QWEN_GDN_DIM; ++i) {
                sk += s[(size_t)i * QWEN_GDN_DIM + j] * kh[i];
            }
            d[j] = (vh[j] - sk) * p->beta[h];
        }

        for (int i = 0; i < QWEN_GDN_DIM; ++i) {
            const float ki = kh[i];
            float *row = s + (size_t)i * QWEN_GDN_DIM;
            for (int j = 0; j < QWEN_GDN_DIM; ++j) {
                row[j] += ki * d[j];
            }
        }

        for (int j = 0; j < QWEN_GDN_DIM; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < QWEN_GDN_DIM; ++i) {
                sum += s[(size_t)i * QWEN_GDN_DIM + j] * qh[i];
            }
            oh[j] = sum;
        }
    }
    return 0;
}

static QWEN_THREAD_RET qwen_gdn_worker_main(void *opaque) {
    qwen_gdn_worker *w = (qwen_gdn_worker *)opaque;
    qwen_gdn_pool *p = w->pool;
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

        w->rc = qwen_gdn_compute_worker(w);

        qwen_mutex_lock(&p->mutex);
        p->done_workers += 1;
        if (p->done_workers == p->n_threads - 1) {
            qwen_cond_signal(&p->done_cond);
        }
    }
}

QWEN_EXPORT void *qwen_gdn_pool_create(int n_threads) {
    if (n_threads < 1 || n_threads > QWEN_GDN_HEADS) return NULL;

    qwen_gdn_pool *p = (qwen_gdn_pool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_threads = n_threads;

    if (qwen_mutex_init(&p->mutex) != 0) {
        free(p);
        return NULL;
    }
    if (qwen_cond_init(&p->work_cond) != 0) {
        qwen_mutex_destroy(&p->mutex);
        free(p);
        return NULL;
    }
    if (qwen_cond_init(&p->done_cond) != 0) {
        qwen_cond_destroy(&p->work_cond);
        qwen_mutex_destroy(&p->mutex);
        free(p);
        return NULL;
    }

    p->workers = (qwen_gdn_worker *)calloc(
        (size_t)n_threads, sizeof(*p->workers));
    if (!p->workers) {
        qwen_cond_destroy(&p->done_cond);
        qwen_cond_destroy(&p->work_cond);
        qwen_mutex_destroy(&p->mutex);
        free(p);
        return NULL;
    }

    for (int i = 0; i < n_threads; ++i) {
        p->workers[i].pool = p;
        p->workers[i].ith = i;
        if (i > 0 && qwen_thread_create(
                &p->workers[i].thread,
                qwen_gdn_worker_main, &p->workers[i]) != 0) {
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
            free(p);
            return NULL;
        }
    }
    return p;
}

QWEN_EXPORT void qwen_gdn_pool_destroy(void *opaque) {
    qwen_gdn_pool *p = (qwen_gdn_pool *)opaque;
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
    qwen_cond_destroy(&p->done_cond);
    qwen_cond_destroy(&p->work_cond);
    qwen_mutex_destroy(&p->mutex);
    free(p);
}

QWEN_EXPORT int qwen_gdn_pool_step_f32(
        void *opaque,
        float *state,
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        float *out) {
    qwen_gdn_pool *p = (qwen_gdn_pool *)opaque;
    if (!p || !state || !q || !k || !v || !gate || !beta || !out) return 1;

    if (p->n_threads == 1) {
        const int rc = qwen_gdn_ar_step_f32_reference(
            state, q, k, v, gate, beta, out);
        if (rc == 0) p->calls += 1;
        return rc;
    }

    qwen_mutex_lock(&p->mutex);
    p->state = state;
    p->q = q;
    p->k = k;
    p->v = v;
    p->gate = gate;
    p->beta = beta;
    p->out = out;
    p->done_workers = 0;
    for (int i = 0; i < p->n_threads; ++i) p->workers[i].rc = 0;
    p->generation += 1;
    qwen_cond_broadcast(&p->work_cond);
    qwen_mutex_unlock(&p->mutex);

    p->workers[0].rc = qwen_gdn_compute_worker(&p->workers[0]);

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

QWEN_EXPORT int qwen_gdn_pool_threads(void *opaque) {
    qwen_gdn_pool *p = (qwen_gdn_pool *)opaque;
    return p ? p->n_threads : 0;
}

QWEN_EXPORT uint64_t qwen_gdn_pool_calls(void *opaque) {
    qwen_gdn_pool *p = (qwen_gdn_pool *)opaque;
    return p ? p->calls : 0;
}

#ifdef QWEN_GDN_PORTABLE_POOL_SELFTEST
#include <stdio.h>

static uint32_t qwen_gdn_rng_state = 0x13553530u;

static uint32_t qwen_gdn_rng_u32(void) {
    qwen_gdn_rng_state =
        qwen_gdn_rng_state * 1664525u + 1013904223u;
    return qwen_gdn_rng_state;
}

static float qwen_gdn_rng_f32(float scale) {
    const int32_t x = (int32_t)(qwen_gdn_rng_u32() >> 8) % 20001 - 10000;
    return (float)x * (scale / 10000.0f);
}

int main(void) {
    enum { STEPS = 8 };
    const size_t state_n =
        (size_t)QWEN_GDN_HEADS * QWEN_GDN_DIM * QWEN_GDN_DIM;
    const size_t vec_n = (size_t)QWEN_GDN_HEADS * QWEN_GDN_DIM;

    float *initial = (float *)malloc(state_n * sizeof(float));
    float *ref_state = (float *)malloc(state_n * sizeof(float));
    float *cand_state = (float *)malloc(state_n * sizeof(float));
    float *q = (float *)malloc((size_t)STEPS * vec_n * sizeof(float));
    float *k = (float *)malloc((size_t)STEPS * vec_n * sizeof(float));
    float *v = (float *)malloc((size_t)STEPS * vec_n * sizeof(float));
    float *gate =
        (float *)malloc((size_t)STEPS * QWEN_GDN_HEADS * sizeof(float));
    float *beta =
        (float *)malloc((size_t)STEPS * QWEN_GDN_HEADS * sizeof(float));
    float *ref_out = (float *)malloc(vec_n * sizeof(float));
    float *cand_out = (float *)malloc(vec_n * sizeof(float));
    if (!initial || !ref_state || !cand_state || !q || !k || !v ||
        !gate || !beta || !ref_out || !cand_out) return 10;

    for (size_t i = 0; i < state_n; ++i) initial[i] = qwen_gdn_rng_f32(0.02f);
    for (int s = 0; s < STEPS; ++s) {
        for (size_t i = 0; i < vec_n; ++i) {
            q[(size_t)s * vec_n + i] = qwen_gdn_rng_f32(0.05f);
            k[(size_t)s * vec_n + i] = qwen_gdn_rng_f32(0.05f);
            v[(size_t)s * vec_n + i] = qwen_gdn_rng_f32(0.20f);
        }
        for (int h = 0; h < QWEN_GDN_HEADS; ++h) {
            gate[(size_t)s * QWEN_GDN_HEADS + h] =
                -0.02f - fabsf(qwen_gdn_rng_f32(0.08f));
            beta[(size_t)s * QWEN_GDN_HEADS + h] =
                0.1f + fabsf(qwen_gdn_rng_f32(0.8f));
        }
    }

    memcpy(ref_state, initial, state_n * sizeof(float));
    for (int s = 0; s < STEPS; ++s) {
        if (qwen_gdn_ar_step_f32_reference(
                ref_state,
                q + (size_t)s * vec_n,
                k + (size_t)s * vec_n,
                v + (size_t)s * vec_n,
                gate + (size_t)s * QWEN_GDN_HEADS,
                beta + (size_t)s * QWEN_GDN_HEADS,
                ref_out) != 0) {
            return 11;
        }
    }

    const int counts[] = {1, 2, 4};
    for (int ci = 0; ci < 3; ++ci) {
        void *pool = qwen_gdn_pool_create(counts[ci]);
        if (!pool) return 20 + ci;
        memcpy(cand_state, initial, state_n * sizeof(float));
        memset(cand_out, 0, vec_n * sizeof(float));

        for (int s = 0; s < STEPS; ++s) {
            if (qwen_gdn_pool_step_f32(
                    pool,
                    cand_state,
                    q + (size_t)s * vec_n,
                    k + (size_t)s * vec_n,
                    v + (size_t)s * vec_n,
                    gate + (size_t)s * QWEN_GDN_HEADS,
                    beta + (size_t)s * QWEN_GDN_HEADS,
                    cand_out) != 0) {
                qwen_gdn_pool_destroy(pool);
                return 30 + ci;
            }
        }

        if (memcmp(ref_state, cand_state, state_n * sizeof(float)) != 0 ||
            memcmp(ref_out, cand_out, vec_n * sizeof(float)) != 0) {
            qwen_gdn_pool_destroy(pool);
            return 40 + ci;
        }
        if (qwen_gdn_pool_threads(pool) != counts[ci] ||
            qwen_gdn_pool_calls(pool) != STEPS) {
            qwen_gdn_pool_destroy(pool);
            return 50 + ci;
        }
        qwen_gdn_pool_destroy(pool);
    }

    free(initial);
    free(ref_state);
    free(cand_state);
    free(q);
    free(k);
    free(v);
    free(gate);
    free(beta);
    free(ref_out);
    free(cand_out);

    puts("QWEN38_GDN_PORTABLE_POOL_BITWISE_PASS");
    return 0;
}
#endif
