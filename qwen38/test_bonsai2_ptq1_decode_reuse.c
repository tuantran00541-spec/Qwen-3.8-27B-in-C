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


static inline __m128i qwen_signed_byte_digit16(
        __m128i raw, int stage) {
    __m128i v = raw;
    if (stage == 1) {
        v = _mm_add_epi8(raw, _mm_add_epi8(raw, raw));
    } else if (stage == 2) {
        v = _mm_add_epi8(
            _mm_and_si128(
                _mm_slli_epi16(raw, 3),
                _mm_set1_epi8((char)-8)),
            raw);
    } else if (stage == 3) {
        const __m128i x3 =
            _mm_add_epi8(raw, _mm_add_epi8(raw, raw));
        v = _mm_add_epi8(
            _mm_and_si128(
                _mm_slli_epi16(x3, 3),
                _mm_set1_epi8((char)-8)),
            x3);
    } else if (stage == 4) {
        const __m128i x9 = _mm_add_epi8(
            _mm_and_si128(
                _mm_slli_epi16(raw, 3),
                _mm_set1_epi8((char)-8)),
            raw);
        v = _mm_add_epi8(
            _mm_and_si128(
                _mm_slli_epi16(x9, 3),
                _mm_set1_epi8((char)-8)),
            x9);
    }

    v = _mm_subs_epu8(v, _mm_set1_epi8(1));
    v = _mm_avg_epu8(
        v, _mm_avg_epu8(v, _mm_setzero_si128()));
    v = _mm_and_si128(
        _mm_srli_epi16(v, 6),
        _mm_set1_epi8(3));
    return _mm_sub_epi8(v, _mm_set1_epi8(1));
}

static inline int32_t qwen_signed_byte_dot16(
        __m128i q8, const int8_t *activation) {
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    const __m128i a8 =
        _mm_loadu_si128((const __m128i *)activation);
    const __m256i a16 = _mm256_cvtepi8_epi16(a8);
    return qwen_hsum8_i32(_mm256_madd_epi16(q16, a16));
}

static inline int32_t qwen_signed_byte_dot8(
        __m128i q8, const int8_t *activation) {
    const __m128i q16 = _mm_cvtepi8_epi16(q8);
    const __m128i a8 =
        _mm_loadl_epi64((const __m128i *)activation);
    const __m128i a16 = _mm_cvtepi8_epi16(a8);
    return qwen_bonsai2_hsum4_i32_sse(
        _mm_madd_epi16(q16, a16));
}

static QWEN_NOINLINE void signed_byte_candidate_block(
        const uint8_t *xb,
        const uint8_t *yb,
        const int8_t lut[256][5],
        int32_t dots[4]) {
    const uint8_t *qs = xb;
    const uint8_t *qh = xb + 24;
    const int8_t *a0 =
        (const int8_t *)(yb + 0 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a1 =
        (const int8_t *)(yb + 1 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a2 =
        (const int8_t *)(yb + 2 * QWEN_BLOCK_Q8_0 + 2);
    const int8_t *a3 =
        (const int8_t *)(yb + 3 * QWEN_BLOCK_Q8_0 + 2);

    const __m128i raw16 =
        _mm_loadu_si128((const __m128i *)qs);
    const __m128i q0 = qwen_signed_byte_digit16(raw16, 0);
    const __m128i q1 = qwen_signed_byte_digit16(raw16, 1);
    const __m128i q2 = qwen_signed_byte_digit16(raw16, 2);
    const __m128i q3 = qwen_signed_byte_digit16(raw16, 3);
    const __m128i q4 = qwen_signed_byte_digit16(raw16, 4);

    const __m128i raw8 =
        _mm_loadl_epi64((const __m128i *)(qs + 16));
    const __m128i q5 = qwen_signed_byte_digit16(raw8, 0);
    const __m128i q6 = qwen_signed_byte_digit16(raw8, 1);
    const __m128i q7 = qwen_signed_byte_digit16(raw8, 2);
    const __m128i q8 = qwen_signed_byte_digit16(raw8, 3);
    const __m128i q9 = qwen_signed_byte_digit16(raw8, 4);

    dots[0] =
        qwen_signed_byte_dot16(q0, a0 + 0) +
        qwen_signed_byte_dot16(q1, a0 + 16);
    dots[1] =
        qwen_signed_byte_dot16(q2, a1 + 0) +
        qwen_signed_byte_dot16(q3, a1 + 16);
    dots[2] =
        qwen_signed_byte_dot16(q4, a2 + 0) +
        qwen_signed_byte_dot8(q5, a2 + 16) +
        qwen_signed_byte_dot8(q6, a2 + 24);
    dots[3] =
        qwen_signed_byte_dot8(q7, a3 + 0) +
        qwen_signed_byte_dot8(q8, a3 + 8) +
        qwen_signed_byte_dot8(q9, a3 + 16);

    for (int nn = 0; nn < 4; ++nn) {
        for (int hh = 0; hh < 2; ++hh) {
            dots[3] +=
                (int32_t)lut[qh[hh]][nn] *
                (int32_t)a3[24 + nn * 2 + hh];
        }
    }
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

            int32_t ref[4], cand[4], signed_cand[4];
            reference_block(xb, yb, lut, ref);
            candidate_block(xb, yb, lut, cand);
            signed_byte_candidate_block(
                xb, yb, lut, signed_cand);
            if (memcmp(ref, cand, sizeof(ref)) != 0) {
                fprintf(stderr,
                    "reuse mismatch mode=%d trial=%d ref=[%d,%d,%d,%d] cand=[%d,%d,%d,%d]\n",
                    mode, trial,
                    ref[0], ref[1], ref[2], ref[3],
                    cand[0], cand[1], cand[2], cand[3]);
                return 1;
            }
            if (memcmp(ref, signed_cand, sizeof(ref)) != 0) {
                fprintf(stderr,
                    "signed-byte mismatch mode=%d trial=%d "
                    "ref=[%d,%d,%d,%d] cand=[%d,%d,%d,%d]\n",
                    mode, trial,
                    ref[0], ref[1], ref[2], ref[3],
                    signed_cand[0], signed_cand[1],
                    signed_cand[2], signed_cand[3]);
                return 2;
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
    enum { FIXTURES = 64, REPS = 800000, SAMPLES = 6 };
    uint8_t xb[FIXTURES][QWEN_BLOCK_PTQ1_0];
    uint8_t yb[FIXTURES][4 * QWEN_BLOCK_Q8_0];
    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    for (int f = 0; f < FIXTURES; ++f) {
        for (size_t i = 0; i < sizeof(xb[f]); ++i) {
            xb[f][i] = (uint8_t)rng_u32();
        }
        fill_activation_values(yb[f], 4);
    }

    volatile int32_t sink = 0;
    int32_t dots[4];
    double production[SAMPLES];
    double signed_byte[SAMPLES];

    for (int sample = 0; sample < SAMPLES; ++sample) {
        for (int slot = 0; slot < 2; ++slot) {
            const int which = (sample + slot) & 1;
            const double t0 = now_s();
            if (which == 0) {
                for (int i = 0; i < REPS; ++i) {
                    const int f = i & (FIXTURES - 1);
                    candidate_block(xb[f], yb[f], lut, dots);
                    sink ^= dots[i & 3];
                }
                production[sample] = now_s() - t0;
            } else {
                for (int i = 0; i < REPS; ++i) {
                    const int f = i & (FIXTURES - 1);
                    signed_byte_candidate_block(
                        xb[f], yb[f], lut, dots);
                    sink ^= dots[i & 3];
                }
                signed_byte[sample] = now_s() - t0;
            }
        }
    }

    double prod_sum = 0.0;
    double signed_sum = 0.0;
    for (int i = 0; i < SAMPLES; ++i) {
        prod_sum += production[i];
        signed_sum += signed_byte[i];
    }
    const double prod_mean = prod_sum / (double)SAMPLES;
    const double signed_mean = signed_sum / (double)SAMPLES;

    printf(
        "QWEN38_BONSAI2_PTQ1_SIGNED_BYTE_BENCH "
        "production_mean_seconds=%.9f "
        "candidate_mean_seconds=%.9f speedup=%.4fx sink=%d\n",
        prod_mean, signed_mean,
        signed_mean > 0.0 ? prod_mean / signed_mean : 0.0,
        (int)sink);
    printf(
        "QWEN38_BONSAI2_PTQ1_SIGNED_BYTE_SAMPLES "
        "production=[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f] "
        "candidate=[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f]\n",
        production[0], production[1], production[2],
        production[3], production[4], production[5],
        signed_byte[0], signed_byte[1], signed_byte[2],
        signed_byte[3], signed_byte[4], signed_byte[5]);
}

int main(void) {
    if (parity() != 0) return 1;
    bench();
    puts("QWEN38_BONSAI2_PTQ1_SIGNED_BYTE_BITWISE_PASS");
    puts("QWEN38_BONSAI2_PTQ1_DECODE_REUSE_BITWISE_PASS");
    return 0;
}
