#include <stddef.h>
#include <stdint.h>

/*
 * Exact-path elementwise residual add for Qwen3.8 staged prefill.
 *
 * Inputs are materialized F32 values. The current Python contract is
 *   f32(f32(a) + f32(b))
 * which is one correctly rounded binary32 addition for binary32 inputs.
 * No reduction or reassociation is involved.
 */
int qwen38_residual_add_many_exact_f32(
        const float *a,
        const float *b,
        size_t n_rows,
        size_t width,
        float *out) {
    if (!a || !b || !out || n_rows == 0 || width == 0) {
        return -1;
    }
    if (n_rows > SIZE_MAX / width) {
        return -2;
    }

    const size_t n = n_rows * width;
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
    return 0;
}
