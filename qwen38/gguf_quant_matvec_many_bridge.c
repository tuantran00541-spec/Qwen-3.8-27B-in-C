/* Export bridge for the exact multi-vector Qwen3.8 prefill kernels.
 *
 * The research implementation keeps its helpers static so the synthetic
 * selftest can include the proven single-vector bridge in one translation
 * unit.  This file deliberately exports only the two many-vector entry points
 * needed by Python real-model gates.  No arithmetic is changed here.
 */
#include <stddef.h>
#include <stdint.h>

#include "gguf_quant_matvec_many_avx2.c"

int qwen_matvec_many_q8_0_q8_0_bridge(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    return qwen_matvec_many_q8_0_q8_0_exact(
        weights, weights_bytes, rows, n,
        activations, activation_bytes_each, n_vec, out);
}

int qwen_matvec_many_q6_k_q8_k_bridge(
        const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n,
        const uint8_t *activations, size_t activation_bytes_each, size_t n_vec,
        float *out) {
    return qwen_matvec_many_q6_k_q8_k_exact(
        weights, weights_bytes, rows, n,
        activations, activation_bytes_each, n_vec, out);
}
