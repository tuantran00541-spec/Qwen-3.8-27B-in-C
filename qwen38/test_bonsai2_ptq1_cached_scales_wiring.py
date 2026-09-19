#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qwen38" / "bonsai2_quant_dot_avx2.c"


def public_ptq1_body(source: str) -> str:
    start = source.index("int qwen_bonsai2_matvec_ptq1_0_q8_0(")
    end = source.index("/* Prism applies optional element signs", start)
    return source[start:end]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = public_ptq1_body(source)
    calls = body.count("qwen_bonsai2_matvec_ptq1_0_q8_0_cached_scales(")
    if calls != 1:
        raise AssertionError(
            f"public PTQ1 path cached-scale calls={calls} expected=1"
        )
    if "qwen_bonsai2_ptq1_lut(lut);" in body:
        raise AssertionError("public PTQ1 path still owns uncached row loop")
    print("QWEN38_BONSAI2_PTQ1_CACHED_SCALES_WIRING_PASS")


if __name__ == "__main__":
    main()
