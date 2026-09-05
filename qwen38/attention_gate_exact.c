#include <math.h>
#include <stddef.h>

static inline float qwen38_f32_round(float x) {
    volatile float y = x;
    return y;
}

int qwen38_attention_gate_exact_f32(
    const float *pregate,
    const float *gate,
    size_t n,
    float *out
) {
    if (pregate == NULL || gate == NULL || out == NULL || n == 0) {
        return -1;
    }

    const float one = qwen38_f32_round(1.0f);
    for (size_t i = 0; i < n; ++i) {
        const float x = qwen38_f32_round(gate[i]);
        const float neg = qwen38_f32_round(-x);
        const float e = qwen38_f32_round(expf(neg));
        const float denom = qwen38_f32_round(one + e);
        const float sig = qwen38_f32_round(one / denom);
        const float p = qwen38_f32_round(pregate[i]);
        out[i] = qwen38_f32_round(p * sig);
    }
    return 0;
}
