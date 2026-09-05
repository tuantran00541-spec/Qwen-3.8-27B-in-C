#include <math.h>
#include <stddef.h>

/*
 * Exact scalar RMSNorm for the pinned Qwen3.8/ggml evidence path.
 *
 * Arithmetic contract mirrored from qwen35_k3_full64_ggml_rmsnorm.py:
 *   - input/value/weight are F32;
 *   - each square is rounded to F32;
 *   - square accumulation is serial double precision (ggml_float);
 *   - mean, eps add, sqrt result, reciprocal scale and each output multiply
 *     are rounded to F32 at the same boundaries as the Python reference.
 *
 * Do not compile this translation unit with fast-math, reassociation, FMA
 * contraction, or tree vectorization.  The serial reduction order is part of
 * the bitwise-exact contract.
 */

static inline float qwen38_f32(float x) {
    volatile float y = x;
    return y;
}

static int qwen38_rmsnorm_row(
    const float *x,
    const float *weight,
    size_t width,
    float eps,
    float *out
) {
    if (x == NULL || weight == NULL || out == NULL || width == 0) {
        return -1;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < width; ++i) {
        const float value = qwen38_f32(x[i]);
        const float square = qwen38_f32(value * value);
        sum_sq += (double)square;
    }

    const float mean = qwen38_f32((float)(sum_sq / (double)width));
    const float eps_f32 = qwen38_f32(eps);
    const float mean_eps = qwen38_f32(mean + eps_f32);

    /* Python reference uses f32(math.sqrt(mean_eps)); doing the sqrt in
       double then rounding once to F32 mirrors that boundary exactly. */
    const float root = qwen38_f32((float)sqrt((double)mean_eps));
    const float scale = qwen38_f32((float)(1.0 / (double)root));

    for (size_t i = 0; i < width; ++i) {
        const float value = qwen38_f32(x[i]);
        const float w = qwen38_f32(weight[i]);
        const float scaled = qwen38_f32(value * scale);
        out[i] = qwen38_f32(scaled * w);
    }
    return 0;
}

int qwen38_rmsnorm_exact_f32(
    const float *x,
    const float *weight,
    size_t width,
    float eps,
    float *out
) {
    return qwen38_rmsnorm_row(x, weight, width, eps, out);
}

int qwen38_rmsnorm_heads_exact_f32(
    const float *values,
    size_t heads,
    size_t head_dim,
    const float *weight,
    float eps,
    float *out
) {
    if (values == NULL || weight == NULL || out == NULL || heads == 0 || head_dim == 0) {
        return -1;
    }
    for (size_t h = 0; h < heads; ++h) {
        const size_t base = h * head_dim;
        const int rc = qwen38_rmsnorm_row(
            values + base, weight, head_dim, eps, out + base);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}
