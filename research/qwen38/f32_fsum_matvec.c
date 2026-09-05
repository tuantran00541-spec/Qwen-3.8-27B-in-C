/* Finite-input F32-weight x F64-activation matvec using CPython-style fsum.
 *
 * The current Qwen3.8 exact research runtime decodes F32 matrices into Python
 * floats and evaluates each row with math.fsum(weight[i] * x[i]).  This probe
 * moves exactly that summation structure to C.  It deliberately avoids SIMD,
 * FMA, fast-math, or reassociation; the first goal is double-bitwise identity
 * with Python math.fsum on the model's finite-value domain.
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
            const double t = x; x = y; y = t;
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
        if (n > 0 && ((lo < 0.0 && s->p[n-1] < 0.0) ||
                      (lo > 0.0 && s->p[n-1] > 0.0))) {
            const double y = lo * 2.0;
            const double x = hi + y;
            const double yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}

static int qwen_dot_f32_f64_fsum_exact(
        const float *weights, const double *x, size_t n, double *out) {
    if (!weights || !x || !out || n == 0) return -1;
    qwen_fsum_partials s;
    qwen_fsum_init(&s);
    int rc = 0;
    for (size_t i = 0; i < n; ++i) {
        const double term = (double)weights[i] * x[i];
        rc = qwen_fsum_add(&s, term);
        if (rc != 0) break;
    }
    if (rc == 0) *out = qwen_fsum_finish(&s);
    qwen_fsum_destroy(&s);
    return rc;
}

int qwen_matvec_f32_fsum_exact(
        const float *weights, size_t rows, size_t n,
        const double *x, double *out) {
    if (!weights || !x || !out || rows == 0 || n == 0) return -1;
    for (size_t r = 0; r < rows; ++r) {
        const int rc = qwen_dot_f32_f64_fsum_exact(weights + r*n, x, n, &out[r]);
        if (rc != 0) return rc;
    }
    return 0;
}
