/* Exact head-parallel Qwen3.8 Gated-DeltaNet autoregressive state probe.
 *
 * Token order remains strictly serial.  Only the 48 independent value-head
 * state planes are partitioned across threads.  Every operation inside one
 * head is copied from the proven gdn_state_ar.c kernel, preserving its exact
 * scalar accumulation order.  This first proof uses pthread create/join per
 * call; promotion should use a persistent cross-platform pool if real-model
 * profiling proves the gain material.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define qwen_gdn_ar_step_f32 qwen_gdn_ar_step_f32_reference
#include "gdn_state_ar.c"
#undef qwen_gdn_ar_step_f32

typedef struct {
    float *state;
    const float *q;
    const float *k;
    const float *v;
    const float *gate;
    const float *beta;
    float *out;
    int h_begin;
    int h_end;
} qwen_gdn_head_task;

static void qwen_gdn_run_heads(qwen_gdn_head_task *t) {
    for (int h = t->h_begin; h < t->h_end; ++h) {
        float *s = t->state + (size_t)h * QWEN_GDN_DIM * QWEN_GDN_DIM;
        const float *qh = t->q + (size_t)h * QWEN_GDN_DIM;
        const float *kh = t->k + (size_t)h * QWEN_GDN_DIM;
        const float *vh = t->v + (size_t)h * QWEN_GDN_DIM;
        float *oh = t->out + (size_t)h * QWEN_GDN_DIM;
        const float decay = expf(t->gate[h]);

        for (int i = 0; i < QWEN_GDN_DIM * QWEN_GDN_DIM; ++i) s[i] *= decay;

        float d[QWEN_GDN_DIM];
        for (int j = 0; j < QWEN_GDN_DIM; ++j) {
            float sk = 0.0f;
            for (int i = 0; i < QWEN_GDN_DIM; ++i) {
                sk += s[(size_t)i * QWEN_GDN_DIM + j] * kh[i];
            }
            d[j] = (vh[j] - sk) * t->beta[h];
        }

        for (int i = 0; i < QWEN_GDN_DIM; ++i) {
            const float ki = kh[i];
            float *row = s + (size_t)i * QWEN_GDN_DIM;
            for (int j = 0; j < QWEN_GDN_DIM; ++j) row[j] += ki * d[j];
        }

        for (int j = 0; j < QWEN_GDN_DIM; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < QWEN_GDN_DIM; ++i) {
                sum += s[(size_t)i * QWEN_GDN_DIM + j] * qh[i];
            }
            oh[j] = sum;
        }
    }
}

static void *qwen_gdn_head_worker(void *opaque) {
    qwen_gdn_run_heads((qwen_gdn_head_task *)opaque);
    return NULL;
}

int qwen_gdn_ar_step_f32_head_parallel(
        float *state,
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        float *out,
        int n_threads) {
    if (!state || !q || !k || !v || !gate || !beta || !out || n_threads < 1) return 1;
    if (n_threads > QWEN_GDN_HEADS) n_threads = QWEN_GDN_HEADS;
    if (n_threads == 1) {
        return qwen_gdn_ar_step_f32_reference(state, q, k, v, gate, beta, out);
    }

    pthread_t *threads = (pthread_t *)calloc((size_t)n_threads, sizeof(*threads));
    qwen_gdn_head_task *tasks = (qwen_gdn_head_task *)calloc((size_t)n_threads, sizeof(*tasks));
    if (!threads || !tasks) {
        free(threads); free(tasks); return 2;
    }

    int created = 0;
    for (int t = 0; t < n_threads; ++t) {
        const int begin = (QWEN_GDN_HEADS * t) / n_threads;
        const int end = (QWEN_GDN_HEADS * (t + 1)) / n_threads;
        tasks[t] = (qwen_gdn_head_task){
            .state = state, .q = q, .k = k, .v = v,
            .gate = gate, .beta = beta, .out = out,
            .h_begin = begin, .h_end = end,
        };
        if (pthread_create(&threads[t], NULL, qwen_gdn_head_worker, &tasks[t]) != 0) {
            for (int j = 0; j < created; ++j) pthread_join(threads[j], NULL);
            free(threads); free(tasks); return 3;
        }
        ++created;
    }
    int rc = 0;
    for (int t = 0; t < created; ++t) {
        if (pthread_join(threads[t], NULL) != 0) rc = 4;
    }
    free(threads); free(tasks);
    return rc;
}

#ifdef QWEN_GDN_HEAD_PARALLEL_SELFTEST
static uint32_t rng_state = 0x13553530u;
static uint32_t rng_u32(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f32(float scale) {
    const int32_t x = (int32_t)(rng_u32() >> 8) % 20001 - 10000;
    return (float)x * (scale / 10000.0f);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { STEPS = 16 };
    const size_t state_n = (size_t)QWEN_GDN_HEADS * QWEN_GDN_DIM * QWEN_GDN_DIM;
    const size_t vec_n = (size_t)QWEN_GDN_HEADS * QWEN_GDN_DIM;
    float *initial = (float *)malloc(state_n * sizeof(float));
    float *ref_state = (float *)malloc(state_n * sizeof(float));
    float *cand_state = (float *)malloc(state_n * sizeof(float));
    float *q = (float *)malloc((size_t)STEPS * vec_n * sizeof(float));
    float *k = (float *)malloc((size_t)STEPS * vec_n * sizeof(float));
    float *v = (float *)malloc((size_t)STEPS * vec_n * sizeof(float));
    float *gate = (float *)malloc((size_t)STEPS * QWEN_GDN_HEADS * sizeof(float));
    float *beta = (float *)malloc((size_t)STEPS * QWEN_GDN_HEADS * sizeof(float));
    float *ref_out = (float *)malloc(vec_n * sizeof(float));
    float *cand_out = (float *)malloc(vec_n * sizeof(float));
    if (!initial || !ref_state || !cand_state || !q || !k || !v || !gate || !beta || !ref_out || !cand_out) return 10;

    for (size_t i = 0; i < state_n; ++i) initial[i] = rng_f32(0.02f);
    for (int s = 0; s < STEPS; ++s) {
        for (size_t i = 0; i < vec_n; ++i) {
            q[(size_t)s*vec_n+i] = rng_f32(0.05f);
            k[(size_t)s*vec_n+i] = rng_f32(0.05f);
            v[(size_t)s*vec_n+i] = rng_f32(0.20f);
        }
        for (int h = 0; h < QWEN_GDN_HEADS; ++h) {
            gate[(size_t)s*QWEN_GDN_HEADS+h] = -0.02f - fabsf(rng_f32(0.08f));
            beta[(size_t)s*QWEN_GDN_HEADS+h] = 0.1f + fabsf(rng_f32(0.8f));
        }
    }

    /* Build one exact serial final-state/output oracle outside all timings. */
    memcpy(ref_state, initial, state_n*sizeof(float));
    for (int s = 0; s < STEPS; ++s) {
        if (qwen_gdn_ar_step_f32_reference(
                ref_state,
                q+(size_t)s*vec_n, k+(size_t)s*vec_n, v+(size_t)s*vec_n,
                gate+(size_t)s*QWEN_GDN_HEADS, beta+(size_t)s*QWEN_GDN_HEADS,
                ref_out) != 0) return 11;
    }

    const int thread_counts[] = {1, 2, 4};
    double elapsed[3] = {0};
    for (int tc = 0; tc < 3; ++tc) {
        memcpy(cand_state, initial, state_n*sizeof(float));
        memset(cand_out, 0, vec_n*sizeof(float));
        const double t0 = now_s();
        for (int s = 0; s < STEPS; ++s) {
            if (qwen_gdn_ar_step_f32_head_parallel(
                    cand_state,
                    q+(size_t)s*vec_n, k+(size_t)s*vec_n, v+(size_t)s*vec_n,
                    gate+(size_t)s*QWEN_GDN_HEADS, beta+(size_t)s*QWEN_GDN_HEADS,
                    cand_out, thread_counts[tc]) != 0) return 12;
        }
        elapsed[tc] = now_s() - t0;
        if (memcmp(ref_state, cand_state, state_n*sizeof(float)) != 0 ||
            memcmp(ref_out, cand_out, vec_n*sizeof(float)) != 0) {
            fprintf(stderr, "GDN mismatch threads=%d\n", thread_counts[tc]);
            return 13;
        }
    }

    printf("GDN head-parallel t1=%.6f t2=%.6f t4=%.6f speedup4=%.4fx\n",
           elapsed[0], elapsed[1], elapsed[2], elapsed[0]/elapsed[2]);
    puts("QWEN38_GDN_HEAD_PARALLEL_BITWISE_PASS");
    free(initial); free(ref_state); free(cand_state); free(q); free(k); free(v);
    free(gate); free(beta); free(ref_out); free(cand_out);
    return 0;
}
#endif
