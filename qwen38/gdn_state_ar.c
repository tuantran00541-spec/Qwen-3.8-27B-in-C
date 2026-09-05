#include <math.h>
#include <stddef.h>
#include <stdint.h>

// Qwen3.8 recurrent dimensions fixed by the pinned decoder contract.
#define QWEN_GDN_HEADS 48
#define QWEN_GDN_DIM 128

// state layout: [head][k_dim][v_dim], row-major in the two 128 dimensions.
// q/k/v are already in 48-head tiled order; q must already include 1/sqrt(128).
// gate contains the pre-exp negative decay values from ssm_a * softplus(...).
// This follows pinned llama.cpp build_delta_net_autoregressive:
//   S = exp(g) * S
//   d = beta * (v - S^T k)
//   S = S + k outer d
//   o = S^T q
int qwen_gdn_ar_step_f32(
        float * state,
        const float * q,
        const float * k,
        const float * v,
        const float * gate,
        const float * beta,
        float * out) {
    if (!state || !q || !k || !v || !gate || !beta || !out) return 1;

    for (int h = 0; h < QWEN_GDN_HEADS; ++h) {
        float * s = state + (size_t) h * QWEN_GDN_DIM * QWEN_GDN_DIM;
        const float * qh = q + (size_t) h * QWEN_GDN_DIM;
        const float * kh = k + (size_t) h * QWEN_GDN_DIM;
        const float * vh = v + (size_t) h * QWEN_GDN_DIM;
        float * oh = out + (size_t) h * QWEN_GDN_DIM;
        const float decay = expf(gate[h]);

        // Decay is a scalar per value head for Qwen3.8 GDA.
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
            float * row = s + (size_t) i * QWEN_GDN_DIM;
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

#ifdef QWEN_GDN_STATE_SELFTEST
#include <stdio.h>
#include <string.h>
int main(void) {
    static float state[QWEN_GDN_HEADS * QWEN_GDN_DIM * QWEN_GDN_DIM];
    static float q[QWEN_GDN_HEADS * QWEN_GDN_DIM];
    static float k[QWEN_GDN_HEADS * QWEN_GDN_DIM];
    static float v[QWEN_GDN_HEADS * QWEN_GDN_DIM];
    static float gate[QWEN_GDN_HEADS];
    static float beta[QWEN_GDN_HEADS];
    static float out[QWEN_GDN_HEADS * QWEN_GDN_DIM];
    memset(state, 0, sizeof(state));
    for (int h = 0; h < QWEN_GDN_HEADS; ++h) {
        gate[h] = 0.0f; beta[h] = 0.5f;
        q[h*QWEN_GDN_DIM] = 0.25f;
        k[h*QWEN_GDN_DIM] = 0.5f;
        v[h*QWEN_GDN_DIM] = 2.0f;
    }
    if (qwen_gdn_ar_step_f32(state,q,k,v,gate,beta,out) != 0) return 2;
    // state[0,0] = .5 * (beta*2)=.5; output0=.5*.25=.125
    if (fabsf(out[0] - 0.125f) > 1e-7f) return 3;
    puts("QWEN_GDN_STATE_SELFTEST PASS");
    return 0;
}
#endif
