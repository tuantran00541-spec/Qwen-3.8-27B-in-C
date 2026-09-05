#include <math.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Exact-path Qwen3.8 GDN depthwise causal conv + SiLU.
 *
 * Arithmetic intentionally mirrors qwen38_k3_prompt_block_many_probe.py:
 *   cur = f32(qkv[t,c] * w[c,3])
 *   cur = f32(cur + f32(prev1[c] * w[c,2]))
 *   cur = f32(cur + f32(prev2[c] * w[c,1]))
 *   cur = f32(cur + f32(prev3[c] * w[c,0]))
 *   out = f32(cur * sigmoid_f32(cur))
 *
 * No FMA/reassociation is permitted by the build flags used by the gate.
 * Initial history is oldest -> newest and may contain 0..3 rows.
 */

static inline float qwen_sigmoid_f32_exact(float x) {
    const float e = expf(-x);
    const float denom = 1.0f + e;
    return 1.0f / denom;
}

static inline const float * qwen_prior_row(
        const float * qkv,
        size_t t,
        size_t conv_dim,
        const float * history,
        size_t history_rows,
        size_t lag) {
    if (lag <= t) {
        return qkv + (t - lag) * conv_dim;
    }
    const size_t need = lag - t;
    if (need > history_rows) {
        return NULL;
    }
    return history + (history_rows - need) * conv_dim;
}

int qwen_gdn_conv_silu_many_f32_exact(
        const float * qkv,
        size_t n_tokens,
        size_t conv_dim,
        const float * kernel4,
        const float * history,
        size_t history_rows,
        float * out) {
    if (!qkv || !kernel4 || !out || conv_dim == 0 || n_tokens == 0 || history_rows > 3) {
        return -1;
    }
    if (history_rows && !history) {
        return -2;
    }

    for (size_t t = 0; t < n_tokens; ++t) {
        const float * cur_row = qkv + t * conv_dim;
        float * out_row = out + t * conv_dim;
        const float * p1 = qwen_prior_row(qkv, t, conv_dim, history, history_rows, 1);
        const float * p2 = qwen_prior_row(qkv, t, conv_dim, history, history_rows, 2);
        const float * p3 = qwen_prior_row(qkv, t, conv_dim, history, history_rows, 3);

        for (size_t c = 0; c < conv_dim; ++c) {
            const float * w = kernel4 + c * 4;
            float acc = cur_row[c] * w[3];
            if (p1) {
                const float term = p1[c] * w[2];
                acc = acc + term;
            }
            if (p2) {
                const float term = p2[c] * w[1];
                acc = acc + term;
            }
            if (p3) {
                const float term = p3[c] * w[0];
                acc = acc + term;
            }
            const float sig = qwen_sigmoid_f32_exact(acc);
            out_row[c] = acc * sig;
        }
    }
    return 0;
}
