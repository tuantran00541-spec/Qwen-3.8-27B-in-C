/* Exact three-pass Qwen3.8 Gated-DeltaNet state probe.
 *
 * Keep the two cache-friendly traversals from the proven kernel unchanged:
 *   1) row-major decay
 *   2) column-wise S^T k reduction for delta
 * Then fuse the row-major outer update with the later S^T q output reduction.
 * For every output column j, q contributions are still accumulated in exact
 * i=0..127 order.  The candidate removes only the separate strided output pass.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define qwen_gdn_ar_step_f32 qwen_gdn_ar_step_f32_reference
#include "gdn_state_ar.c"
#undef qwen_gdn_ar_step_f32

int qwen_gdn_ar_step_f32_fuse_output_exact(
        float *state,
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        float *out) {
    if (!state || !q || !k || !v || !gate || !beta || !out) return 1;

    for (int h = 0; h < QWEN_GDN_HEADS; ++h) {
        float *s = state + (size_t)h * QWEN_GDN_DIM * QWEN_GDN_DIM;
        const float *qh = q + (size_t)h * QWEN_GDN_DIM;
        const float *kh = k + (size_t)h * QWEN_GDN_DIM;
        const float *vh = v + (size_t)h * QWEN_GDN_DIM;
        float *oh = out + (size_t)h * QWEN_GDN_DIM;
        const float decay = expf(gate[h]);

        /* Pass 1: byte-for-byte same row-major decay traversal. */
        for (int i = 0; i < QWEN_GDN_DIM * QWEN_GDN_DIM; ++i) s[i] *= decay;

        /* Pass 2: byte-for-byte same j-major delta reduction. */
        float d[QWEN_GDN_DIM];
        for (int j = 0; j < QWEN_GDN_DIM; ++j) {
            float sk = 0.0f;
            for (int i = 0; i < QWEN_GDN_DIM; ++i) {
                sk += s[(size_t)i * QWEN_GDN_DIM + j] * kh[i];
            }
            d[j] = (vh[j] - sk) * beta[h];
            oh[j] = 0.0f;
        }

        /* Pass 3: same row-major state update, plus output accumulation.
         * For a fixed j, iterations reach oh[j] in i=0..127 order exactly as
         * the old separate output loop did. */
        for (int i = 0; i < QWEN_GDN_DIM; ++i) {
            const float ki = kh[i];
            const float qi = qh[i];
            float *row = s + (size_t)i * QWEN_GDN_DIM;
            for (int j = 0; j < QWEN_GDN_DIM; ++j) {
                row[j] += ki * d[j];
                oh[j] += row[j] * qi;
            }
        }
    }
    return 0;
}

#ifdef QWEN_GDN_FUSE_OUTPUT_SELFTEST
static uint32_t rng_state = 0x353038f0u;
static uint32_t rng_u32(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f32(float scale) {
    const int32_t x = (int32_t)(rng_u32() >> 8) % 20001 - 10000;
    return (float)x * (scale / 10000.0f);
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { STEPS = 64 };
    const size_t state_n = (size_t)QWEN_GDN_HEADS * QWEN_GDN_DIM * QWEN_GDN_DIM;
    const size_t vec_n = (size_t)QWEN_GDN_HEADS * QWEN_GDN_DIM;
    float *initial=(float *)malloc(state_n*sizeof(float));
    float *ref_state=(float *)malloc(state_n*sizeof(float));
    float *cand_state=(float *)malloc(state_n*sizeof(float));
    float *q=(float *)malloc((size_t)STEPS*vec_n*sizeof(float));
    float *k=(float *)malloc((size_t)STEPS*vec_n*sizeof(float));
    float *v=(float *)malloc((size_t)STEPS*vec_n*sizeof(float));
    float *gate=(float *)malloc((size_t)STEPS*QWEN_GDN_HEADS*sizeof(float));
    float *beta=(float *)malloc((size_t)STEPS*QWEN_GDN_HEADS*sizeof(float));
    float *ref_out=(float *)malloc(vec_n*sizeof(float));
    float *cand_out=(float *)malloc(vec_n*sizeof(float));
    if(!initial||!ref_state||!cand_state||!q||!k||!v||!gate||!beta||!ref_out||!cand_out) return 10;

    for(size_t i=0;i<state_n;++i) initial[i]=rng_f32(0.03f);
    for(int s=0;s<STEPS;++s){
        for(size_t i=0;i<vec_n;++i){
            q[(size_t)s*vec_n+i]=rng_f32(0.06f);
            k[(size_t)s*vec_n+i]=rng_f32(0.06f);
            v[(size_t)s*vec_n+i]=rng_f32(0.25f);
        }
        for(int h=0;h<QWEN_GDN_HEADS;++h){
            gate[(size_t)s*QWEN_GDN_HEADS+h]=-0.01f-fabsf(rng_f32(0.10f));
            beta[(size_t)s*QWEN_GDN_HEADS+h]=0.05f+fabsf(rng_f32(0.90f));
        }
    }

    memcpy(ref_state,initial,state_n*sizeof(float));
    double t0=now_s();
    for(int s=0;s<STEPS;++s){
        if(qwen_gdn_ar_step_f32_reference(ref_state,q+(size_t)s*vec_n,k+(size_t)s*vec_n,v+(size_t)s*vec_n,
                gate+(size_t)s*QWEN_GDN_HEADS,beta+(size_t)s*QWEN_GDN_HEADS,ref_out)!=0) return 11;
    }
    const double ref_s=now_s()-t0;

    memcpy(cand_state,initial,state_n*sizeof(float));
    t0=now_s();
    for(int s=0;s<STEPS;++s){
        if(qwen_gdn_ar_step_f32_fuse_output_exact(cand_state,q+(size_t)s*vec_n,k+(size_t)s*vec_n,v+(size_t)s*vec_n,
                gate+(size_t)s*QWEN_GDN_HEADS,beta+(size_t)s*QWEN_GDN_HEADS,cand_out)!=0) return 12;
    }
    const double cand_s=now_s()-t0;

    if(memcmp(ref_state,cand_state,state_n*sizeof(float))!=0){fprintf(stderr,"fuse-output state mismatch\n");return 13;}
    if(memcmp(ref_out,cand_out,vec_n*sizeof(float))!=0){fprintf(stderr,"fuse-output output mismatch\n");return 14;}
    printf("GDN four-pass=%.6f three-pass=%.6f speedup=%.4fx\n",ref_s,cand_s,ref_s/cand_s);
    puts("QWEN38_GDN_FUSE_OUTPUT_BITWISE_PASS");
    free(initial);free(ref_state);free(cand_state);free(q);free(k);free(v);free(gate);free(beta);free(ref_out);free(cand_out);
    return 0;
}
#endif
