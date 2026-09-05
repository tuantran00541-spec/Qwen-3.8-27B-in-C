/* Exact full-attention causal core for the Qwen3.8 research runtime.
 *
 * This kernel intentionally mirrors the current Python arithmetic instead of
 * using flash-attention algebra:
 *   score = f32(math.fsum(float(q[d]) * float(k[d])) * scale)
 *   softmax uses libm expf, F32 sequential denominator accumulation
 *   V accumulation keeps the original temporal F32 mul/add order
 *
 * Compile without fast-math, reassociation, or FMA contraction.  The goal is
 * bitwise identity with the existing Python path, not approximate equivalence.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define QWEN_FSUM_STACK_PARTIALS 32

typedef struct {
    double stack[QWEN_FSUM_STACK_PARTIALS];
    double *p;
    size_t n;
    size_t cap;
} qwen_fsum_partials;

static void qwen_fsum_init(qwen_fsum_partials *s) {
    s->p = s->stack;
    s->n = 0;
    s->cap = QWEN_FSUM_STACK_PARTIALS;
}

static void qwen_fsum_destroy(qwen_fsum_partials *s) {
    if (s->p != s->stack) free(s->p);
    s->p = s->stack;
    s->n = 0;
    s->cap = QWEN_FSUM_STACK_PARTIALS;
}

static int qwen_fsum_grow(qwen_fsum_partials *s) {
    const size_t new_cap = s->cap * 2;
    if (new_cap <= s->cap || new_cap > SIZE_MAX / sizeof(double)) return -1;
    double *next;
    if (s->p == s->stack) {
        next = (double *)malloc(new_cap * sizeof(double));
        if (!next) return -1;
        for (size_t i = 0; i < s->n; ++i) next[i] = s->stack[i];
    } else {
        next = (double *)realloc(s->p, new_cap * sizeof(double));
        if (!next) return -1;
    }
    s->p = next;
    s->cap = new_cap;
    return 0;
}

static int qwen_fsum_add(qwen_fsum_partials *s, double x) {
    if (!isfinite(x)) return -2;
    size_t i = 0;
    for (size_t j = 0; j < s->n; ++j) {
        double y = s->p[j];
        if (fabs(x) < fabs(y)) {
            const double t = x;
            x = y;
            y = t;
        }
        const double hi = x + y;
        const double yr = hi - x;
        const double lo = y - yr;
        if (lo != 0.0) s->p[i++] = lo;
        x = hi;
    }
    s->n = i;
    if (x != 0.0) {
        if (s->n >= s->cap && qwen_fsum_grow(s) != 0) return -3;
        s->p[s->n++] = x;
    }
    return 0;
}

static double qwen_fsum_finish(qwen_fsum_partials *s) {
    double hi = 0.0;
    double lo = 0.0;
    if (s->n > 0) {
        size_t n = s->n;
        hi = s->p[--n];
        while (n > 0) {
            const double x = hi;
            const double y = s->p[--n];
            hi = x + y;
            const double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && s->p[n - 1] < 0.0) ||
                      (lo > 0.0 && s->p[n - 1] > 0.0))) {
            const double y = lo * 2.0;
            const double x = hi + y;
            const double yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}

static int qwen_dot_f32_f32_fsum_exact(
        const float *a, const float *b, size_t n, double *out) {
    qwen_fsum_partials s;
    qwen_fsum_init(&s);
    int rc = 0;
    for (size_t i = 0; i < n; ++i) {
        const double term = (double)a[i] * (double)b[i];
        rc = qwen_fsum_add(&s, term);
        if (rc != 0) break;
    }
    if (rc == 0) *out = qwen_fsum_finish(&s);
    qwen_fsum_destroy(&s);
    return rc;
}

static inline float qwen_addf(float a, float b) {
    return (float)((double)a + (double)b);
}

static inline float qwen_mulf(float a, float b) {
    return (float)((double)a * (double)b);
}

/*
 * q:       [q_heads, head_dim], F32 values already normalized/RoPE'd.
 * k_cache: [n_ctx, kv_heads, head_dim], F16-cache values widened exactly to F32.
 * v_cache: same layout as k_cache.
 * out:     [q_heads, head_dim], current Python "pregate" ordering.
 */
int qwen_attention_core_f32_exact(
        const float *q,
        size_t q_heads,
        size_t kv_heads,
        size_t head_dim,
        const float *k_cache,
        const float *v_cache,
        size_t n_ctx,
        double scale,
        float *out) {
    if (!q || !k_cache || !v_cache || !out ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || n_ctx == 0 ||
        q_heads % kv_heads != 0 || !isfinite(scale)) {
        return -1;
    }

    float *scores = (float *)malloc(n_ctx * sizeof(float));
    float *probs = (float *)malloc(n_ctx * sizeof(float));
    if (!scores || !probs) {
        free(scores);
        free(probs);
        return -4;
    }

    const size_t repeat = q_heads / kv_heads;
    const size_t kv_dim = kv_heads * head_dim;

    for (size_t qh = 0; qh < q_heads; ++qh) {
        const size_t kvh = qh / repeat;
        const float *qv = q + qh * head_dim;

        for (size_t ti = 0; ti < n_ctx; ++ti) {
            const float *kh = k_cache + ti * kv_dim + kvh * head_dim;
            double dot = 0.0;
            const int rc = qwen_dot_f32_f32_fsum_exact(qv, kh, head_dim, &dot);
            if (rc != 0) {
                free(scores);
                free(probs);
                return rc;
            }
            scores[ti] = (float)(dot * scale);
        }

        /* Python max() keeps the first equal value.  This comparison does too. */
        float max_score = scores[0];
        for (size_t ti = 1; ti < n_ctx; ++ti) {
            if (scores[ti] > max_score) max_score = scores[ti];
        }

        float denom = 0.0f;
        for (size_t ti = 0; ti < n_ctx; ++ti) {
            const float diff = (float)((double)scores[ti] - (double)max_score);
            const float e = expf(diff);
            probs[ti] = e;
            denom = qwen_addf(denom, e);
        }
        if (!(denom > 0.0f) || !isfinite(denom)) {
            free(scores);
            free(probs);
            return -5;
        }
        for (size_t ti = 0; ti < n_ctx; ++ti) {
            probs[ti] = (float)((double)probs[ti] / (double)denom);
        }

        for (size_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (size_t ti = 0; ti < n_ctx; ++ti) {
                const float vv = v_cache[ti * kv_dim + kvh * head_dim + d];
                const float prod = qwen_mulf(probs[ti], vv);
                acc = qwen_addf(acc, prod);
            }
            out[qh * head_dim + d] = acc;
        }
    }

    free(scores);
    free(probs);
    return 0;
}
