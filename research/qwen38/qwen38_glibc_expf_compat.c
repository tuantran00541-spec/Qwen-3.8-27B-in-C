/*
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 *
 * Diagnostic x86-64 expf compatibility core for the Qwen3.8 Win32 gate.
 * The range-reduction algorithm, polynomial coefficients, and exp2 table are
 * independently adapted from Arm Optimized Routines' permissively licensed
 * scalar expf/exp2f data.  The goal is to reproduce the glibc x86-64 FMA expf
 * value semantics without changing the pinned Linux decoder arithmetic.
 *
 * Overflow/underflow return values are reproduced; errno/fenv side effects are
 * intentionally outside this model-value diagnostic contract.
 */
#include <math.h>
#include <stdint.h>

/* If a future Windows build redirects model expf calls with -Dexpf=..., keep
 * this translation unit's own declaration/body independent of that macro. */
#ifdef expf
#undef expf
#endif

#define QWEN38_EXPF_TABLE_BITS 5
#define QWEN38_EXPF_N (1u << QWEN38_EXPF_TABLE_BITS)

static const uint64_t qwen38_expf_tab[QWEN38_EXPF_N] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL,
    0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL,
    0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL,
    0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL,
    0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL,
    0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};

static inline uint64_t qwen38_as_u64(double x) {
    union {
        double f;
        uint64_t i;
    } u = {x};
    return u.i;
}

static inline double qwen38_as_f64(uint64_t x) {
    union {
        uint64_t i;
        double f;
    } u = {x};
    return u.f;
}

static inline double qwen38_fma(double a, double b, double c) {
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_fma(a, b, c);
#else
    return fma(a, b, c);
#endif
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
float qwen38_glibc_expf_compat(float x) {
    /* Match the value side of the Arm/glibc special cases. */
    if (x != x) {
        return x + x;
    }
    if (x == -INFINITY) {
        return 0.0f;
    }
    if (x == INFINITY) {
        return x;
    }
    if (x > 0x1.62e42ep6f) {
        return INFINITY;
    }
    if (x < -0x1.9fe368p6f) {
        return 0.0f;
    }

    const double inv_ln2_n = 0x1.71547652b82fep+0 * 32.0;
    const double shift = 0x1.8p+52;
    const double c0 = 0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0;
    const double c1 = 0x1.ebfce50fac4f3p-3 / 32.0 / 32.0;
    const double c2 = 0x1.62e42ff0c52d6p-1 / 32.0;

    const double xd = (double)x;
    const double z0 = inv_ln2_n * xd;
    double kd = z0 + shift;
    const uint64_t ki = qwen38_as_u64(kd);
    kd -= shift;
    const double r = z0 - kd;

    uint64_t t = qwen38_expf_tab[ki % QWEN38_EXPF_N];
    t += ki << (52 - QWEN38_EXPF_TABLE_BITS);
    const double s = qwen38_as_f64(t);

    const double z = qwen38_fma(c0, r, c1);
    const double r2 = r * r;
    double y = qwen38_fma(c2, r, 1.0);
    y = qwen38_fma(z, r2, y);
    y *= s;
    return (float)y;
}
