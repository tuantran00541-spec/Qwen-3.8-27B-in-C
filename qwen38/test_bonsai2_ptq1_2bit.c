/* RED/GREEN exactness + microbenchmark gate for an expanded 2-bit
 * lane-major PTQ1 runtime layout.
 *
 * The production PTQ1_0 encoding is 28 bytes / 128 weights.  The wished-for
 * runtime layout is 34 bytes / 128 weights: 32 bytes of 2-bit ternary codes
 * plus the original fp16 scale.  Within each Q8_0 group of 32 weights, byte j
 * stores weights j, 8+j, 16+j and 24+j in bit pairs 0,2,4,6.  That layout lets
 * the dot kernel recover eight contiguous ternary values with one shift/mask.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bonsai2_quant_dot_avx2.c"

static uint32_t rng_state = 0x13552b17u;
static volatile float sink_value = 0.0f;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static uint16_t random_f16_scale(void) {
    static const uint16_t scales[] = {
        0x3000u, 0x3400u, 0x3800u, 0x3c00u, 0x4000u,
    };
    return scales[rng_u32() % (sizeof(scales) / sizeof(scales[0]))];
}

static uint32_t f32_bits(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static double now_seconds(void) {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_ptq1(uint8_t *w, size_t n) {
    const size_t nb = n / QWEN_QK_PTQ1_0;
    for (size_t ib = 0; ib < nb; ++ib) {
        uint8_t *b = w + ib * QWEN_BLOCK_PTQ1_0;
        for (int i = 0; i < 26; ++i) b[i] = (uint8_t)rng_u32();
        const uint16_t d = random_f16_scale();
        b[26] = (uint8_t)(d & 0xffu);
        b[27] = (uint8_t)(d >> 8);
    }
}

static void fill_q8(uint8_t *a, size_t n) {
    const size_t nb = n / QWEN_QK8_0;
    for (size_t ib = 0; ib < nb; ++ib) {
        uint8_t *b = a + ib * QWEN_BLOCK_Q8_0;
        const uint16_t d = random_f16_scale();
        b[0] = (uint8_t)(d & 0xffu);
        b[1] = (uint8_t)(d >> 8);
        for (int i = 0; i < 32; ++i) {
            b[2 + i] = (uint8_t)((int)(rng_u32() % 255u) - 127);
        }
    }
}

static int one_exact_case(size_t n, int trial) {
    const size_t src_bytes =
        (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t dst_bytes = qwen_bonsai2_ptq1_2bit_row_bytes(n);
    const size_t act_bytes =
        (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;

    uint8_t *src = (uint8_t *)malloc(src_bytes);
    uint8_t *dst = (uint8_t *)malloc(dst_bytes);
    uint8_t *act = (uint8_t *)malloc(act_bytes);
    if (!src || !dst || !act) return 90;

    fill_ptq1(src, n);
    fill_q8(act, n);

    const int rc = qwen_bonsai2_expand_ptq1_2bit(
        src, src_bytes, n, dst, dst_bytes);
    if (rc != 0) {
        fprintf(stderr, "expand rc=%d n=%zu trial=%d\n", rc, n, trial);
        return 91;
    }

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    const float ref = qwen_bonsai2_vec_dot_ptq1_q8_0(src, act, n, lut);
    const float got = qwen_bonsai2_vec_dot_ptq1_2bit_q8_0(dst, act, n);

    free(src);
    free(dst);
    free(act);

    if (f32_bits(ref) != f32_bits(got)) {
        fprintf(stderr,
                "2bit mismatch n=%zu trial=%d ref=%.9g got=%.9g "
                "ref_bits=%08x got_bits=%08x\n",
                n, trial, ref, got, f32_bits(ref), f32_bits(got));
        return 1;
    }
    return 0;
}

static int benchmark(void) {
    const size_t n = 5120;
    const size_t rows = 2048;
    const int repeats = 12;
    const size_t src_row =
        (n / QWEN_QK_PTQ1_0) * QWEN_BLOCK_PTQ1_0;
    const size_t dst_row = qwen_bonsai2_ptq1_2bit_row_bytes(n);
    const size_t act_bytes =
        (n / QWEN_QK8_0) * QWEN_BLOCK_Q8_0;

    uint8_t *src = (uint8_t *)malloc(rows * src_row);
    uint8_t *dst = (uint8_t *)malloc(rows * dst_row);
    uint8_t *act = (uint8_t *)malloc(act_bytes);
    if (!src || !dst || !act) return 92;

    for (size_t r = 0; r < rows; ++r) fill_ptq1(src + r * src_row, n);
    fill_q8(act, n);
    for (size_t r = 0; r < rows; ++r) {
        const int rc = qwen_bonsai2_expand_ptq1_2bit(
            src + r * src_row, src_row, n,
            dst + r * dst_row, dst_row);
        if (rc != 0) return 93;
    }

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);

    float sum_ref = 0.0f;
    double t0 = now_seconds();
    for (int rep = 0; rep < repeats; ++rep) {
        for (size_t r = 0; r < rows; ++r) {
            sum_ref += qwen_bonsai2_vec_dot_ptq1_q8_0(
                src + r * src_row, act, n, lut);
        }
    }
    const double ref_s = now_seconds() - t0;

    float sum_new = 0.0f;
    t0 = now_seconds();
    for (int rep = 0; rep < repeats; ++rep) {
        for (size_t r = 0; r < rows; ++r) {
            sum_new += qwen_bonsai2_vec_dot_ptq1_2bit_q8_0(
                dst + r * dst_row, act, n);
        }
    }
    const double new_s = now_seconds() - t0;
    sink_value = sum_ref + sum_new;

    printf("QWEN38_BONSAI2_PTQ1_2BIT_BENCH "
           "rows=%zu n=%zu repeats=%d ref_seconds=%.9f "
           "candidate_seconds=%.9f speedup=%.4fx "
           "storage_ratio=%.6f\n",
           rows, n, repeats, ref_s, new_s, ref_s / new_s,
           (double)dst_row / (double)src_row);

    free(src);
    free(dst);
    free(act);
    return 0;
}

int main(void) {
    static const size_t widths[] = {128, 256, 1024, 5120, 17408};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); ++wi) {
        for (int trial = 0; trial < 32; ++trial) {
            const int rc = one_exact_case(widths[wi], trial);
            if (rc != 0) return rc;
        }
    }
    puts("QWEN38_BONSAI2_PTQ1_2BIT_BITWISE_PASS");
    return benchmark();
}
