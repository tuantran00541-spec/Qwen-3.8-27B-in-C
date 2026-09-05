/* Exact single-thread AVX2 overlay for the proven Qwen3.8 scalar bridge.
 *
 * Goal: vectorize integer inner products without changing FP accumulation
 * order.  Activation quantization stays on the proven scalar path.  The
 * public matvec ABI is unchanged, so the Python runtime can A/B this shared
 * object without any frontend changes.
 */
#include <immintrin.h>

#define qwen_matvec_q8_0_q8_0_scalar qwen_matvec_q8_0_q8_0_reference
#define qwen_matvec_q6_k_q8_k_scalar qwen_matvec_q6_k_q8_k_reference
#define qwen_vec_dot_q8_0_q8_0_scalar qwen_vec_dot_q8_0_q8_0_reference
#define qwen_vec_dot_q6_k_q8_k_scalar qwen_vec_dot_q6_k_q8_k_reference
#include "gguf_quant_dot.c"
#undef qwen_matvec_q8_0_q8_0_scalar
#undef qwen_matvec_q6_k_q8_k_scalar
#undef qwen_vec_dot_q8_0_q8_0_scalar
#undef qwen_vec_dot_q6_k_q8_k_scalar

static inline int32_t qwen_hsum8_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

float qwen_vec_dot_q8_0_q8_0_scalar(const uint8_t *x, size_t x_bytes, const uint8_t *y, size_t y_bytes, size_t n) {
    if (!x || !y || n == 0 || n % QWEN_QK8_0 != 0) return NAN;
    const size_t nb = n / QWEN_QK8_0;
    if (x_bytes != nb * QWEN_BLOCK_Q8_0 || y_bytes != nb * QWEN_BLOCK_Q8_0) return NAN;
    float sum = 0.0f;
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = x + ib * QWEN_BLOCK_Q8_0;
        const uint8_t *yb = y + ib * QWEN_BLOCK_Q8_0;
        const __m128i x0 = _mm_loadu_si128((const __m128i *)(xb + 2));
        const __m128i x1 = _mm_loadu_si128((const __m128i *)(xb + 18));
        const __m128i y0 = _mm_loadu_si128((const __m128i *)(yb + 2));
        const __m128i y1 = _mm_loadu_si128((const __m128i *)(yb + 18));
        const __m256i xlo = _mm256_cvtepi8_epi16(x0);
        const __m256i xhi = _mm256_cvtepi8_epi16(x1);
        const __m256i ylo = _mm256_cvtepi8_epi16(y0);
        const __m256i yhi = _mm256_cvtepi8_epi16(y1);
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i p0 = _mm256_madd_epi16(_mm256_mullo_epi16(xlo, ylo), ones);
        __m256i p1 = _mm256_madd_epi16(_mm256_mullo_epi16(xhi, yhi), ones);
        const int32_t sumi = qwen_hsum8_i32(_mm256_add_epi32(p0, p1));
        const float dx = qwen_f16_to_f32(qwen_load_u16_le(xb));
        const float dy = qwen_f16_to_f32(qwen_load_u16_le(yb));
        sum += (float)sumi * dx * dy;
    }
    return sum;
}

float qwen_vec_dot_q6_k_q8_k_scalar(const uint8_t *q6, size_t q6_bytes, const uint8_t *q8k, size_t q8k_bytes, size_t n) {
    if (!q6 || !q8k || n == 0 || n % QWEN_QK_K != 0) return NAN;
    const size_t nb = n / QWEN_QK_K;
    if (q6_bytes != nb * QWEN_BLOCK_Q6_K || q8k_bytes != nb * QWEN_BLOCK_Q8_K) return NAN;
    float sums[8] = {0};
    int8_t aux8[QWEN_QK_K];
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = q6 + ib * QWEN_BLOCK_Q6_K;
        const uint8_t *yb = q8k + ib * QWEN_BLOCK_Q8_K;
        const uint8_t *ql = xb;
        const uint8_t *qh = xb + 128;
        const int8_t *scales = (const int8_t *)(xb + 192);
        const int8_t *q8 = (const int8_t *)(yb + 4);
        int8_t *a = aux8;
        for (int j = 0; j < QWEN_QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l+0]  = (int8_t)(((ql[l+0]&15)  | (((qh[l]>>0)&3)<<4))-32);
                a[l+32] = (int8_t)(((ql[l+32]&15) | (((qh[l]>>2)&3)<<4))-32);
                a[l+64] = (int8_t)(((ql[l+0]>>4)  | (((qh[l]>>4)&3)<<4))-32);
                a[l+96] = (int8_t)(((ql[l+32]>>4) | (((qh[l]>>6)&3)<<4))-32);
            }
            a += 128; ql += 64; qh += 32;
        }
        __m256i lanes = _mm256_setzero_si256();
        a = aux8;
        for (int j = 0; j < QWEN_QK_K/16; ++j) {
            const __m256i scale = _mm256_set1_epi32((int)scales[j]);
            __m256i av = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)a));
            __m256i qv = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)q8));
            lanes = _mm256_add_epi32(lanes, _mm256_mullo_epi32(scale, _mm256_mullo_epi32(av, qv)));
            a += 8; q8 += 8;
            av = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)a));
            qv = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)q8));
            lanes = _mm256_add_epi32(lanes, _mm256_mullo_epi32(scale, _mm256_mullo_epi32(av, qv)));
            a += 8; q8 += 8;
        }
        int32_t lanev[8];
        _mm256_storeu_si256((__m256i *)lanev, lanes);
        const float d = qwen_f16_to_f32(qwen_load_u16_le(xb + 208)) * qwen_load_f32_le(yb);
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)lanev[l];
    }
    float sumf = 0.0f;
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

int qwen_matvec_q8_0_q8_0_scalar(const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n, const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n % QWEN_QK8_0 != 0) return -1;
    const size_t row_bytes=(n/QWEN_QK8_0)*QWEN_BLOCK_Q8_0;
    if (activation_bytes != row_bytes) return -2;
    if (weights_bytes % row_bytes != 0 || weights_bytes/row_bytes != rows) return -3;
    for (size_t r=0;r<rows;++r) out[r]=qwen_vec_dot_q8_0_q8_0_scalar(weights+r*row_bytes,row_bytes,activation,activation_bytes,n);
    return 0;
}

int qwen_matvec_q6_k_q8_k_scalar(const uint8_t *weights, size_t weights_bytes, size_t rows, size_t n, const uint8_t *activation, size_t activation_bytes, float *out) {
    if (!weights || !activation || !out || rows == 0 || n == 0 || n % QWEN_QK_K != 0) return -1;
    const size_t wr=(n/QWEN_QK_K)*QWEN_BLOCK_Q6_K, ar=(n/QWEN_QK_K)*QWEN_BLOCK_Q8_K;
    if (activation_bytes != ar) return -2;
    if (weights_bytes % wr != 0 || weights_bytes/wr != rows) return -3;
    for (size_t r=0;r<rows;++r) out[r]=qwen_vec_dot_q6_k_q8_k_scalar(weights+r*wr,wr,activation,activation_bytes,n);
    return 0;
}

#ifdef QWEN_AVX2_EXACT_SELFTEST
static void avx2_pack_q6_fixture(
        uint8_t out[QWEN_BLOCK_Q6_K], uint16_t f16_scale,
        const int8_t scales[16], const int8_t quants[256]) {
    memset(out, 0, QWEN_BLOCK_Q6_K);
    uint8_t *ql = out;
    uint8_t *qh = out + 128;
    for (int n = 0; n < 256; n += 128) {
        const int ql_base = n == 0 ? 0 : 64;
        const int qh_base = n == 0 ? 0 : 32;
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = (uint8_t)(quants[n+l+0] + 32);
            const uint8_t q2 = (uint8_t)(quants[n+l+32] + 32);
            const uint8_t q3 = (uint8_t)(quants[n+l+64] + 32);
            const uint8_t q4 = (uint8_t)(quants[n+l+96] + 32);
            ql[ql_base+l] = (uint8_t)((q1 & 15) | ((q3 & 15) << 4));
            ql[ql_base+l+32] = (uint8_t)((q2 & 15) | ((q4 & 15) << 4));
            qh[qh_base+l] = (uint8_t)(((q1 >> 4) & 3) | (((q2 >> 4)&3)<<2) | (((q3>>4)&3)<<4) | (((q4>>4)&3)<<6));
        }
    }
    memcpy(out + 192, scales, 16);
    out[208] = (uint8_t)(f16_scale & 0xffu);
    out[209] = (uint8_t)(f16_scale >> 8);
}

int main(void) {
    uint32_t s=0x12345678u;
    for (int trial=0; trial<200; ++trial) {
        uint8_t q6[QWEN_BLOCK_Q6_K], q8k[QWEN_BLOCK_Q8_K];
        int8_t sc[16], qv[256]; float x[256];
        for(int i=0;i<16;++i){s=s*1664525u+1013904223u;sc[i]=(int8_t)((s>>24)%31-15);}
        for(int i=0;i<256;++i){s=s*1664525u+1013904223u;qv[i]=(int8_t)((s>>24)%64-32);s=s*1664525u+1013904223u;x[i]=((int32_t)(s>>8)%20001-10000)/997.0f;}
        avx2_pack_q6_fixture(q6,qwen_f32_to_f16(0.03125f),sc,qv);
        if(qwen_quantize_q8_k_scalar(x,256,q8k,sizeof(q8k))!=0) return 2;
        float a=qwen_vec_dot_q6_k_q8_k_reference(q6,sizeof(q6),q8k,sizeof(q8k),256);
        float b=qwen_vec_dot_q6_k_q8_k_scalar(q6,sizeof(q6),q8k,sizeof(q8k),256);
        if(memcmp(&a,&b,sizeof(float))!=0){fprintf(stderr,"Q6 mismatch trial=%d %.9g %.9g\n",trial,a,b);return 3;}
    }
    for (int trial=0; trial<200; ++trial) {
        float a0[256],a1[256]; uint8_t q0[272],q1[272];
        for(int i=0;i<256;++i){s=s*1664525u+1013904223u;a0[i]=((int32_t)(s>>8)%20001-10000)/701.0f;s=s*1664525u+1013904223u;a1[i]=((int32_t)(s>>8)%20001-10000)/809.0f;}
        qwen_quantize_q8_0_scalar(a0,256,q0,sizeof(q0)); qwen_quantize_q8_0_scalar(a1,256,q1,sizeof(q1));
        float a=qwen_vec_dot_q8_0_q8_0_reference(q0,sizeof(q0),q1,sizeof(q1),256);
        float b=qwen_vec_dot_q8_0_q8_0_scalar(q0,sizeof(q0),q1,sizeof(q1),256);
        if(memcmp(&a,&b,sizeof(float))!=0){fprintf(stderr,"Q8 mismatch trial=%d %.9g %.9g\n",trial,a,b);return 4;}
    }
    puts("QWEN38_AVX2_SINGLE_THREAD_BITWISE_PASS");
    return 0;
}
#endif
