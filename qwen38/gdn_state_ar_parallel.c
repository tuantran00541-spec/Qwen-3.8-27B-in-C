/* OpenMP head-parallel overlay for the proven Qwen3.8 Gated-DeltaNet AR state.
 *
 * Each of the 48 value heads owns a disjoint 128x128 state matrix and output
 * slice.  Parallelizing the outer head loop therefore preserves the exact
 * operation order within each head while allowing independent heads to run on
 * different CPU workers.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define qwen_gdn_ar_step_f32 qwen_gdn_ar_step_f32_serial
#include "gdn_state_ar.c"
#undef qwen_gdn_ar_step_f32

#ifdef _OPENMP
#include <omp.h>
#endif

int qwen_gdn_parallel_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int qwen_gdn_ar_step_f32(
        float *state,
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        float *out) {
    if (!state || !q || !k || !v || !gate || !beta || !out) return 1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < QWEN_GDN_HEADS; ++h) {
        float *s = state + (size_t) h * QWEN_GDN_DIM * QWEN_GDN_DIM;
        const float *qh = q + (size_t) h * QWEN_GDN_DIM;
        const float *kh = k + (size_t) h * QWEN_GDN_DIM;
        const float *vh = v + (size_t) h * QWEN_GDN_DIM;
        float *oh = out + (size_t) h * QWEN_GDN_DIM;
        const float decay = expf(gate[h]);

        for (int i = 0; i < QWEN_GDN_DIM * QWEN_GDN_DIM; ++i) s[i] *= decay;

        float d[QWEN_GDN_DIM];
        for (int j = 0; j < QWEN_GDN_DIM; ++j) {
            float sk = 0.0f;
            for (int i = 0; i < QWEN_GDN_DIM; ++i) {
                sk += s[(size_t) i * QWEN_GDN_DIM + j] * kh[i];
            }
            d[j] = (vh[j] - sk) * beta[h];
        }

        for (int i = 0; i < QWEN_GDN_DIM; ++i) {
            const float ki = kh[i];
            float *row = s + (size_t) i * QWEN_GDN_DIM;
            for (int j = 0; j < QWEN_GDN_DIM; ++j) row[j] += ki * d[j];
        }

        for (int j = 0; j < QWEN_GDN_DIM; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < QWEN_GDN_DIM; ++i) {
                sum += s[(size_t) i * QWEN_GDN_DIM + j] * qh[i];
            }
            oh[j] = sum;
        }
    }
    return 0;
}

#ifdef QWEN_GDN_PARALLEL_SELFTEST
int main(void) {
    enum { E = QWEN_GDN_HEADS * QWEN_GDN_DIM * QWEN_GDN_DIM,
           V = QWEN_GDN_HEADS * QWEN_GDN_DIM };
    static float serial_state[E], parallel_state[E];
    static float q[V], k[V], v[V], gate[QWEN_GDN_HEADS], beta[QWEN_GDN_HEADS];
    static float serial_out[V], parallel_out[V];

    for (int i = 0; i < E; ++i) {
        serial_state[i] = (float) (((i * 13) % 97) - 48) * 0.0001f;
        parallel_state[i] = serial_state[i];
    }
    for (int i = 0; i < V; ++i) {
        q[i] = (float) (((i * 17) % 101) - 50) * 0.001f;
        k[i] = (float) (((i * 19) % 103) - 51) * 0.001f;
        v[i] = (float) (((i * 23) % 107) - 53) * 0.002f;
    }
    for (int h = 0; h < QWEN_GDN_HEADS; ++h) {
        gate[h] = -0.01f * (float) (1 + (h % 7));
        beta[h] = 0.1f + 0.01f * (float) (h % 11);
    }

    if (qwen_gdn_ar_step_f32_serial(serial_state, q, k, v, gate, beta, serial_out) != 0) return 2;
    if (qwen_gdn_ar_step_f32(parallel_state, q, k, v, gate, beta, parallel_out) != 0) return 3;
    if (memcmp(serial_state, parallel_state, sizeof(serial_state)) != 0) return 4;
    if (memcmp(serial_out, parallel_out, sizeof(serial_out)) != 0) return 5;

    printf("QWEN_GDN_OPENMP_HEAD_EXACT_PASS threads=%d heads=%d\n",
           qwen_gdn_parallel_max_threads(), QWEN_GDN_HEADS);
    return 0;
}
#endif
