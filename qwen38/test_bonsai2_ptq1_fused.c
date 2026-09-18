/* RED/GREEN bitwise gate for the fused PTQ1_0 x Q8_0 dot path.
 *
 * This translation unit includes the production kernel so it can compare the
 * existing decode-then-AVX2 reference with the wished-for fused decoder/dot
 * without exposing either helper through the runtime ABI.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bonsai2_quant_dot_avx2.c"

static uint32_t rng_state = 0x13552727u;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static uint16_t random_f16_scale(void) {
    static const uint16_t scales[] = {
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25  */
        0x3800u, /* 0.5   */
        0x3c00u, /* 1.0   */
        0x4000u, /* 2.0   */
    };
    return scales[rng_u32() % (sizeof(scales) / sizeof(scales[0]))];
}

static uint32_t f32_bits(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int one_case(size_t n, int trial) {
    const size_t n_ptq = n / QWEN_QK_PTQ1_0;
    const size_t n_q8 = n / QWEN_QK8_0;
    const size_t wbytes = n_ptq * QWEN_BLOCK_PTQ1_0;
    const size_t abytes = n_q8 * QWEN_BLOCK_Q8_0;
    uint8_t *w = (uint8_t *)malloc(wbytes);
    uint8_t *a = (uint8_t *)malloc(abytes);
    if (!w || !a) return 90;

    for (size_t ib = 0; ib < n_ptq; ++ib) {
        uint8_t *b = w + ib * QWEN_BLOCK_PTQ1_0;
        for (int i = 0; i < 26; ++i) b[i] = (uint8_t)rng_u32();
        const uint16_t d = random_f16_scale();
        b[26] = (uint8_t)(d & 0xffu);
        b[27] = (uint8_t)(d >> 8);
    }
    for (size_t ib = 0; ib < n_q8; ++ib) {
        uint8_t *b = a + ib * QWEN_BLOCK_Q8_0;
        const uint16_t d = random_f16_scale();
        b[0] = (uint8_t)(d & 0xffu);
        b[1] = (uint8_t)(d >> 8);
        for (int i = 0; i < 32; ++i) {
            b[2 + i] = (uint8_t)((int)(rng_u32() % 255u) - 127);
        }
    }

    int8_t lut[256][5];
    qwen_bonsai2_ptq1_lut(lut);
    const float ref = qwen_bonsai2_vec_dot_ptq1_q8_0(w, a, n, lut);
    const float fused = qwen_bonsai2_vec_dot_ptq1_q8_0_fused(w, a, n, lut);

    free(w);
    free(a);

    if (f32_bits(ref) != f32_bits(fused)) {
        fprintf(stderr,
                "PTQ1 fused mismatch n=%zu trial=%d ref=%.9g fused=%.9g "
                "ref_bits=%08x fused_bits=%08x\n",
                n, trial, ref, fused, f32_bits(ref), f32_bits(fused));
        return 1;
    }
    return 0;
}

int main(void) {
    static const size_t widths[] = {128, 256, 1024, 5120, 17408};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); ++wi) {
        for (int trial = 0; trial < 64; ++trial) {
            const int rc = one_case(widths[wi], trial);
            if (rc != 0) return rc;
        }
    }
    puts("QWEN38_BONSAI2_PTQ1_FUSED_BITWISE_PASS");
    return 0;
}
