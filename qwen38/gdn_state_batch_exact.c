#include <stddef.h>

#define QWEN_GDN_HEADS 48
#define QWEN_GDN_DIM 128
#define QWEN_GDN_VALUE_DIM (QWEN_GDN_HEADS * QWEN_GDN_DIM)

/*
 * Keep the proven scalar autoregressive state step authoritative.  This batch
 * ABI only amortizes Python/ctypes call and input-marshaling overhead: rows are
 * executed strictly in increasing order against the same persistent state.
 * No arithmetic is fused, vectorized, or reassociated here.
 */
int qwen_gdn_ar_step_f32(
        float * state,
        const float * q,
        const float * k,
        const float * v,
        const float * gate,
        const float * beta,
        float * out);

int qwen_gdn_ar_batch_f32(
        float * state,
        size_t rows,
        const float * q,
        const float * k,
        const float * v,
        const float * gate,
        const float * beta,
        float * out) {
    if (!state || !q || !k || !v || !gate || !beta || !out) return 1;
    for (size_t row = 0; row < rows; ++row) {
        const size_t vo = row * (size_t) QWEN_GDN_VALUE_DIM;
        const size_t ho = row * (size_t) QWEN_GDN_HEADS;
        const int rc = qwen_gdn_ar_step_f32(
            state,
            q + vo,
            k + vo,
            v + vo,
            gate + ho,
            beta + ho,
            out + vo);
        if (rc != 0) return rc;
    }
    return 0;
}
