#include <stddef.h>

/*
 * Exact sequential batch boundary for Q/K per-head RMSNorm.
 *
 * This translation unit intentionally does not implement RMSNorm arithmetic.
 * It only loops over prompt rows and calls the already-proven exact
 * qwen38_rmsnorm_heads_exact_f32 entry point for each row in original order.
 * Head order and the reduction order inside every head remain unchanged.
 */

int qwen38_rmsnorm_heads_exact_f32(
    const float *values,
    size_t heads,
    size_t head_dim,
    const float *weight,
    float eps,
    float *out
);

int qwen38_rmsnorm_heads_many_exact_f32(
    const float *values,
    size_t rows,
    size_t heads,
    size_t head_dim,
    const float *weight,
    float eps,
    float *out
) {
    if (values == NULL || weight == NULL || out == NULL ||
        rows == 0 || heads == 0 || head_dim == 0) {
        return -1;
    }
    const size_t row_width = heads * head_dim;
    if (row_width / head_dim != heads) {
        return -2;
    }
    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * row_width;
        const int rc = qwen38_rmsnorm_heads_exact_f32(
            values + base, heads, head_dim, weight, eps, out + base);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}
