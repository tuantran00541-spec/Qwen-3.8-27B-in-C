/* Real-gate wrapper: export the existing state ABI through the proven
 * 4-thread head-parallel implementation.  Linux proof only; Windows promotion
 * will replace pthread mechanics with the cross-platform pool after E2E value
 * is established.
 */
#include "gdn_state_head_parallel.c"

int qwen_gdn_ar_step_f32(
        float *state,
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        float *out) {
    return qwen_gdn_ar_step_f32_head_parallel(state, q, k, v, gate, beta, out, 4);
}
