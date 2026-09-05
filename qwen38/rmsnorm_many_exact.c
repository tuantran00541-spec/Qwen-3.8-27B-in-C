#include <stddef.h>

int qwen38_rmsnorm_exact_f32(
    const float *x,
    const float *weight,
    size_t width,
    float eps,
    float *out);

/*
 * Sequential many-row wrapper around the proven exact RMSNorm row kernel.
 * This only amortizes Python/ctypes marshaling and call overhead.  Row order,
 * reduction order, and all F32 materialization boundaries remain unchanged.
 */
int qwen38_rmsnorm_many_exact_f32(
    const float *values,
    size_t rows,
    size_t width,
    const float *weight,
    float eps,
    float *out) {
    if (!values || !weight || !out || rows == 0 || width == 0) return -1;
    for (size_t row = 0; row < rows; ++row) {
        const size_t base = row * width;
        const int rc = qwen38_rmsnorm_exact_f32(
            values + base, weight, width, eps, out + base);
        if (rc != 0) return rc;
    }
    return 0;
}
