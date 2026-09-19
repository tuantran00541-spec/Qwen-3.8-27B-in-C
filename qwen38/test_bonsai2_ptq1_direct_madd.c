#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bonsai2_quant_dot_avx2.c"

static uint32_t rng_state = 0x27b5ca1eu;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static void fill_activation(int8_t *a, size_t n, int mode) {
    for (size_t i = 0; i < n; ++i) {
        switch (mode) {
            case 0: a[i] = (int8_t)-128; break;
            case 1: a[i] = (int8_t)127; break;
            case 2: a[i] = (int8_t)((i & 1u) ? 127 : -128); break;
            case 3: a[i] = (int8_t)((i % 3u) - 1); break;
            default: a[i] = (int8_t)(uint8_t)rng_u32(); break;
        }
    }
}

static int check16(void) {
    static const uint16_t pow3[] = {1, 3, 9, 27, 81};
    uint8_t packed[16];
    int8_t activation[16];

    for (int mode = 0; mode < 5; ++mode) {
        fill_activation(activation, 16, mode);
        for (size_t p = 0; p < sizeof(pow3) / sizeof(pow3[0]); ++p) {
            for (int trial = 0; trial < 512; ++trial) {
                for (int i = 0; i < 16; ++i) packed[i] = (uint8_t)rng_u32();
                const int32_t ref =
                    qwen_bonsai2_ptq1_dot_trits16_avx2(
                        packed, pow3[p], activation);
                const int32_t cand =
                    qwen_bonsai2_ptq1_dot_trits16_madd_avx2(
                        packed, pow3[p], activation);
                if (ref != cand) {
                    fprintf(stderr,
                        "direct-madd16 mismatch mode=%d pow3=%u trial=%d ref=%d cand=%d\n",
                        mode, (unsigned)pow3[p], trial, ref, cand);
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int check8(void) {
    static const uint16_t pow3[] = {1, 3, 9, 27, 81};
    uint8_t packed[8];
    int8_t activation[8];

    for (int mode = 0; mode < 5; ++mode) {
        fill_activation(activation, 8, mode);
        for (size_t p = 0; p < sizeof(pow3) / sizeof(pow3[0]); ++p) {
            for (int trial = 0; trial < 512; ++trial) {
                for (int i = 0; i < 8; ++i) packed[i] = (uint8_t)rng_u32();
                const int32_t ref =
                    qwen_bonsai2_ptq1_dot_trits8_sse(
                        packed, pow3[p], activation);
                const int32_t cand =
                    qwen_bonsai2_ptq1_dot_trits8_madd_sse(
                        packed, pow3[p], activation);
                if (ref != cand) {
                    fprintf(stderr,
                        "direct-madd8 mismatch mode=%d pow3=%u trial=%d ref=%d cand=%d\n",
                        mode, (unsigned)pow3[p], trial, ref, cand);
                    return 2;
                }
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

static int benchmark16(void) {
    enum { REPS = 2000000 };
    uint8_t packed[16];
    int8_t activation[16];
    for (int i = 0; i < 16; ++i) {
        packed[i] = (uint8_t)rng_u32();
        activation[i] = (int8_t)(uint8_t)rng_u32();
    }

    volatile int32_t sink = 0;
    double t0 = now_s();
    for (int i = 0; i < REPS; ++i) {
        sink ^= qwen_bonsai2_ptq1_dot_trits16_avx2(
            packed, (uint16_t)(1u << (i % 5)), activation);
    }
    const double reference_s = now_s() - t0;

    t0 = now_s();
    for (int i = 0; i < REPS; ++i) {
        sink ^= qwen_bonsai2_ptq1_dot_trits16_madd_avx2(
            packed, (uint16_t)(1u << (i % 5)), activation);
    }
    const double candidate_s = now_s() - t0;

    printf(
        "PTQ1 direct madd16 reference=%.6f candidate=%.6f speedup=%.4fx sink=%d\n",
        reference_s, candidate_s,
        candidate_s > 0.0 ? reference_s / candidate_s : 0.0,
        (int)sink);
    return 0;
}

int main(void) {
    if (check16() != 0) return 1;
    if (check8() != 0) return 2;
    if (benchmark16() != 0) return 3;
    puts("QWEN38_BONSAI2_PTQ1_DIRECT_MADD_BITWISE_PASS");
    return 0;
}
