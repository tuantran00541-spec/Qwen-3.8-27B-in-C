#include <stddef.h>

static inline float qwen38_f32_round(float x) {
    volatile float y = x;
    return y;
}

int qwen38_gdn_repeat_scale_many_exact_f32(
    const float *q,
    const float *k,
    size_t rows,
    size_t key_dim,
    size_t repeats,
    float scale,
    float *q_out,
    float *k_out
) {
    if (q == NULL || k == NULL || q_out == NULL || k_out == NULL ||
        rows == 0 || key_dim == 0 || repeats == 0) {
        return -1;
    }

    const float scale_f = qwen38_f32_round(scale);
    const size_t out_width = key_dim * repeats;
    for (size_t row = 0; row < rows; ++row) {
        const float *q_row = q + row * key_dim;
        const float *k_row = k + row * key_dim;
        float *q_dst = q_out + row * out_width;
        float *k_dst = k_out + row * out_width;

        for (size_t rep = 0; rep < repeats; ++rep) {
            const size_t base = rep * key_dim;
            for (size_t i = 0; i < key_dim; ++i) {
                q_dst[base + i] = qwen38_f32_round(
                    qwen38_f32_round(q_row[i]) * scale_f);
                k_dst[base + i] = qwen38_f32_round(k_row[i]);
            }
        }
    }
    return 0;
}
