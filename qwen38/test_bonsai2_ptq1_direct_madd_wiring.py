#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qwen38" / "bonsai2_quant_dot_avx2.c"


def function_body(source: str, signature: str, next_marker: str) -> str:
    start = source.index(signature)
    end = source.index(next_marker, start)
    return source[start:end]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = function_body(
        source,
        "static float qwen_bonsai2_vec_dot_ptq1_q8_0_fused_cached_scales(",
        "static int qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(",
    )
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
