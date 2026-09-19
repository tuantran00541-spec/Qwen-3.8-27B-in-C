#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qwen38" / "bonsai2_quant_dot_avx2.c"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index(
        "static float qwen_bonsai2_vec_dot_ptq1_q8_0_fused_cached_scales("
    )
    end = source.index(
        "static int qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(",
        start,
    )
    body = source[start:end]
    call = "qwen_bonsai2_ptq1_dot_block_reuse_avx2("
    if body.count(call) != 1:
        raise AssertionError(
            f"cached-scale fused PTQ1 block helper calls={body.count(call)} expected=1"
        )
    forbidden = (
        "qwen_bonsai2_ptq1_dot_trits16_madd_avx2(",
        "qwen_bonsai2_ptq1_dot_trits8_madd_sse(",
        "qwen_bonsai2_ptq1_dot_trits16_avx2(",
        "qwen_bonsai2_ptq1_dot_trits8_sse(",
    )
    for needle in forbidden:
        if needle in body:
            raise AssertionError(
                f"cached-scale fused PTQ1 still performs per-segment decode: {needle}"
            )
    print("QWEN38_BONSAI2_PTQ1_DECODE_REUSE_WIRING_PASS")


if __name__ == "__main__":
    main()
