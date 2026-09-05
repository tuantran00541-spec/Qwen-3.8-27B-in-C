#include <math.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Exact-path Qwen3.8 SwiGLU for staged multi-token prefill.
 *
 * Arithmetic intentionally mirrors qwen35_k3_two_token.py / the current
 * qwen38_k3_prompt_block_many_probe.py FFN path:
 *   sigmoid = f32(1.0f / f32(1.0f + expf(-gate)))
 *   silu    = f32(gate * sigmoid)
 *   out     = f32(silu * up)
 *
 * The real gate compiles this without fast-math, reassociation, FMA
 * contraction, or tree vectorization.  Each element is independent; no
 * reduction order is changed.
 */

static inline float qwen_sigmoid_f32_exact(float x) {
    const float e = expf(-x);
    const float denom = 1.0f + e;
    return 1.0f / denom;
}

int qwen_swiglu_many_f32_exact(
        const float * gate,
        const float * up,
        size_t n_rows,
        size_t width,
        float * out) {
    if (!gate || !up || !out || n_rows == 0 || width == 0) {
        return -1;
    }
    if (n_rows > SIZE_MAX / width) {
        return -2;
    }

    const size_t n = n_rows * width;
    for (size_t i = 0; i < n; ++i) {
        const float x = gate[i];
        const float sig = qwen_sigmoid_f32_exact(x);
        const float silu = x * sig;
        out[i] = silu * up[i];
    }
    return 0;
}
