#include <math.h>
#include <stddef.h>

/*
 * Bitwise-oriented Qwen3.8 GDN output RMSNorm + SiLU gate.
 *
 * Contract mirrors the proven Python exact path:
 *   - each square is rounded to F32 before accumulating in double
 *   - sum is accumulated strictly d=0..head_dim-1
 *   - mean/epsilon/sqrt/scale return to F32 at the same boundaries
 *   - SiLU uses expf and F32 denominator/sigmoid/multiply semantics
 *   - no reassociation/FMA/vector reduction is permitted by the build flags
 *
 * The norm weight is shared by every value head and has head_dim entries.
 */

static inline float qwen_sigmoid_f32_exact(float x) {
    const float e = expf(-x);
    const float denom = 1.0f + e;
    return (float)(1.0 / (double)denom);
}

static inline float qwen_silu_f32_exact(float x) {
    const float sig = qwen_sigmoid_f32_exact(x);
    return x * sig;
}

int qwen_gdn_output_rmsnorm_gate_f32_exact(
    const float *core,
    const float *z,
    const float *weight,
    size_t heads,
    size_t head_dim,
    float eps,
    float *out) {
    if (!core || !z || !weight || !out || heads == 0 || head_dim == 0) {
        return -1;
    }

    for (size_t h = 0; h < heads; ++h) {
        const size_t base = h * head_dim;
        double sum_sq = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            const float v = core[base + d];
            const float sq = v * v;
            sum_sq += (double)sq;
        }

        const float mean = (float)(sum_sq / (double)head_dim);
        const float mean_eps = mean + eps;
        const float root = sqrtf(mean_eps);
        const float scale = (float)(1.0 / (double)root);

        for (size_t d = 0; d < head_dim; ++d) {
            const size_t idx = base + d;
            const float norm0 = core[idx] * scale;
            const float norm1 = norm0 * weight[d];
            const float gate = qwen_silu_f32_exact(z[idx]);
            out[idx] = norm1 * gate;
        }
    }
    return 0;
}
