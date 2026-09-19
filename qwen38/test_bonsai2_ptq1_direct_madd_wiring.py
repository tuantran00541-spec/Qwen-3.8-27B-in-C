#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qwen38" / "bonsai2_quant_dot_avx2.c"


def function_body(source: str, signature: str, next_marker: str) -> str:
    start = source.index(signature)
    end = source.index(next_marker, start)
    return source[start:end]


def require_direct_madd_pair_helpers(source: str) -> None:
    pair16 = function_body(
        source,
        "static inline __m256i qwen_bonsai2_ptq1_pairs16_from_raw_avx2(",
        "static inline __m128i qwen_bonsai2_ptq1_pairs8_from_raw_sse(",
    )
    if "_mm256_madd_epi16(q, a16)" not in pair16:
        raise AssertionError("decode-reuse AVX2 path lost direct madd")
    if "_mm256_mullo_epi16(q, a16)" in pair16:
        raise AssertionError("decode-reuse AVX2 path rematerializes q*a before madd")

    pair8 = function_body(
        source,
        "static inline __m128i qwen_bonsai2_ptq1_pairs8_from_raw_sse(",
        "static inline int32_t qwen_bonsai2_hsum4_i32_sse(",
    )
    if "_mm_madd_epi16(q, a16)" not in pair8:
        raise AssertionError("decode-reuse SSE path lost direct madd")
    if "_mm_mullo_epi16(q, a16)" in pair8:
        raise AssertionError("decode-reuse SSE path rematerializes q*a before madd")


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = function_body(
        source,
        "static float qwen_bonsai2_vec_dot_ptq1_q8_0_fused_cached_scales(",
        "static int qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(",
    )

    block_call = "qwen_bonsai2_ptq1_dot_block_reuse_avx2("
    if block_call in body:
        if body.count(block_call) != 1:
            raise AssertionError(
                f"decode-reuse block calls={body.count(block_call)} expected=1"
            )
        block = function_body(
            source,
            "static inline void qwen_bonsai2_ptq1_dot_block_reuse_avx2(",
            "/* Fused PTQ1 decoder + Q8 dot.",
        )
        required = {
            "qwen_bonsai2_ptq1_pairs16_from_raw_avx2(": 5,
            "qwen_bonsai2_ptq1_pairs8_from_raw_sse(": 5,
        }
        for needle, expected in required.items():
            actual = block.count(needle)
            if actual != expected:
                raise AssertionError(
                    f"{needle} calls={actual} expected={expected}"
                )
        require_direct_madd_pair_helpers(source)
    else:
        required = {
            "qwen_bonsai2_ptq1_dot_trits16_madd_avx2(": 5,
            "qwen_bonsai2_ptq1_dot_trits8_madd_sse(": 5,
        }
        for needle, expected in required.items():
            actual = body.count(needle)
            if actual != expected:
                raise AssertionError(
                    f"{needle} calls={actual} expected={expected}"
                )

    forbidden = (
        "qwen_bonsai2_ptq1_dot_trits16_avx2(",
        "qwen_bonsai2_ptq1_dot_trits8_sse(",
    )
    for needle in forbidden:
        if needle in body:
            raise AssertionError(
                f"cached-scale PTQ1 fused path still calls legacy helper {needle}"
            )
    print("QWEN38_BONSAI2_PTQ1_DIRECT_MADD_WIRING_PASS")


if __name__ == "__main__":
    main()
