#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bonsai2_quant_dot_avx2.c"

#if defined(__GNUC__) || defined(__clang__)
#define QWEN_NOINLINE __attribute__((noinline))
#else
#define QWEN_NOINLINE
#endif

static uint32_t rng_state = 0x51a27b3du;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static void fill_activation_values(uint8_t yb[4 * QWEN_BLOCK_Q8_0], int mode) {
    memset(yb, 0, 4 * QWEN_BLOCK_Q8_0);
    for (int k = 0; k < 4; ++k) {
        int8_t *a = (int8_t *)(yb + k * QWEN_BLOCK_Q8_0 + 2);
        for (int i = 0; i < 32; ++i) {
            switch (mode) {
                case 0: a[i] = (int8_t)-128; break;
                case 1: a[i] = (int8_t)127; break;
                case 2: a[i] = (int8_t)(((i + k) & 1) ? 127 : -128); break;
                case 3: a[i] = (int8_t)((i % 3) - 1); break;
                default: a[i] = (int8_t)(uint8_t)rng_u32(); break;
            }
        }
    }
}

static QWEN_NOINLINE void reference_block(
        const uint8_t *xb,
        const uint8_t *yb,
        const int8_t lut[256][5],
        int32_t dots[4]) {
    static const uint16_t pow3[5] = {1, 3, 9, 27, 81};
    const uint8_t *qs = xb;
    const uint8_t *qh = xb + 24;
    const int8_t *a0 = (const int8_t *)(yb + 0 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a1 = (const int8_t *)(yb + 1 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a2 = (const int8_t *)(yb + 2 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a3 = (const int8_t *)(yb + 3 * QWEN_BLOCK_Q8_0 + 2);

    dots[0] = 0; dots[1] = 0; dots[2] = 0; dots[3] = 0;
    dots[0] += qwen_bonsai2_ptq1_dot_trits16_madd_avx2(qs, pow3[0], a0 + 0);
    dots[0] += qwen_bonsai2_ptq1_dot_trits16_madd_avx2(qs, pow3[1], a0 + 16);
    dots[1] += qwen_bonsai2_ptq1_dot_trits16_madd_avx2(qs, pow3[2], a1 + 0);
    dots[1] += qwen_bonsai2_ptq1_dot_trits16_madd_avx2(qs, pow3[3], a1 + 16);
    dots[2] += qwen_bonsai2_ptq1_dot_trits16_madd_avx2(qs, pow3[4], a2 + 0);

    dots[2] += qwen_bonsai2_ptq1_dot_trits8_madd_sse(qs + 16, pow3[0], a2 + 16);
    dots[2] += qwen_bonsai2_ptq1_dot_trits8_madd_sse(qs + 16, pow3[1], a2 + 24);
    dots[3] += qwen_bonsai2_ptq1_dot_trits8_madd_sse(qs + 16, pow3[2], a3 + 0);
    dots[3] += qwen_bonsai2_ptq1_dot_trits8_madd_sse(qs + 16, pow3[3], a3 + 8);
    dots[3] += qwen_bonsai2_ptq1_dot_trits8_madd_sse(qs + 16, pow3[4], a3 + 16);

    for (int nn = 0; nn < 4; ++nn) {
        for (int hh = 0; hh < 2; ++hh) {
            dots[3] +=
                (int32_t)lut[qh[hh]][nn] *
                (int32_t)a3[24 + nn * 2 + hh];
        }
    }
}

static QWEN_NOINLINE void candidate_block(
        const uint8_t *xb,
        const uint8_t *yb,
        const int8_t lut[256][5],
        int32_t dots[4]) {
    qwen_bonsai2_ptq1_dot_block_reuse_avx2(xb, yb, lut, dots);
}

static int parity(void) {
    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);

    for (int mode = 0; mode < 5; ++mode) {
        for (int trial = 0; trial < 2048; ++trial) {
            uint8_t xb[QWEN_BLOCK_PTQ1_0];
            uint8_t yb[4 * QWEN_BLOCK_Q8_0];
            for (size_t i = 0; i < sizeof(xb); ++i) xb[i] = (uint8_t)rng_u32();
            fill_activation_values(yb, mode);

            int32_t ref[4], cand[4];
            reference_block(xb, yb, lut, ref);
            candidate_block(xb, yb, lut, cand);
            if (memcmp(ref, cand, sizeof(ref)) != 0) {
                fprintf(stderr,
                    "reuse mismatch mode=%d trial=%d ref=[%d,%d,%d,%d] cand=[%d,%d,%d,%d]\n",
                    mode, trial,
                    ref[0], ref[1], ref[2], ref[3],
                    cand[0], cand[1], cand[2], cand[3]);
                return 1;
            }
        }
    }
    return 0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench(void) {
    enum { FIXTURES = 64, REPS = 800000 };
    uint8_t xb[FIXTURES][QWEN_BLOCK_PTQ1_0];
    uint8_t yb[FIXTURES][4 * QWEN_BLOCK_Q8_0];
    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    for (int f = 0; f < FIXTURES; ++f) {
        for (size_t i = 0; i < sizeof(xb[f]); ++i) xb[f][i] = (uint8_t)rng_u32();
        fill_activation_values(yb[f], 4);
    }

    volatile int32_t sink = 0;
    int32_t dots[4];
    double t0 = now_s();
    for (int i = 0; i < REPS; ++i) {
        const int f = i & (FIXTURES - 1);
        reference_block(xb[f], yb[f], lut, dots);
        sink ^= dots[i & 3];
    }
    const double ref_s = now_s() - t0;

    t0 = now_s();
    for (int i = 0; i < REPS; ++i) {
        const int f = i & (FIXTURES - 1);
        candidate_block(xb[f], yb[f], lut, dots);
        sink ^= dots[i & 3];
    }
    const double cand_s = now_s() - t0;

    printf("PTQ1 decode-reuse reference=%.6f candidate=%.6f speedup=%.4fx sink=%d\n",
           ref_s, cand_s, cand_s > 0.0 ? ref_s / cand_s : 0.0, (int)sink);
}

int main(void) {
    if (parity() != 0) return 1;
    bench();
    puts("QWEN38_BONSAI2_PTQ1_DECODE_REUSE_BITWISE_PASS");
    return 0;
}
