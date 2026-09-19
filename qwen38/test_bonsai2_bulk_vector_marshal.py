#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from test_bonsai2_bulk_matvec_marshal import check_helper

SOURCE = ROOT / "qwen38" / "bonsai2_quant_runtime.py"


def method_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"    def {name}(")
    end = source.index(f"    def {next_name}(", start)
    return source[start:end]


def require_bulk(
    source: str,
    name: str,
    next_name: str,
    expected: str,
    forbidden: str,
) -> None:
    body = method_body(source, name, next_name)
    calls = body.count(expected)
    if calls != 1:
        raise AssertionError(
            f"{name}: bulk output marshal calls={calls} expected=1"
        )
    if forbidden in body:
        raise AssertionError(
            f"{name}: scalar Python output copy still wired"
        )


def main() -> None:
    check_helper()
    source = SOURCE.read_text(encoding="utf-8")
    cases = (
        ("swiglu", "attention_sigmoid_mul",
         "self._marshal_f32_output(out, n)",
         "[float(out[i]) for i in range(n)]"),
        ("attention_sigmoid_mul", "_array_f32_ptr",
         "self._marshal_f32_output(out, n)",
         "[float(out[i]) for i in range(n)]"),
        ("attention_core", "gdn_repeat_scale",
         "self._marshal_f32_output(out, q_dim)",
         "[float(out[i]) for i in range(q_dim)]"),
        ("residual_add", "gdn_conv_silu",
         "self._marshal_f32_output(out, n)",
         "[float(out[i]) for i in range(n)]"),
        ("gdn_conv_silu", "gdn_norm_gate",
         "self._marshal_f32_output(out, n)",
         "[float(out[i]) for i in range(n)]"),
        ("gdn_norm_gate", "rms_norm",
         "self._marshal_f32_output(out, n)",
         "[float(out[i]) for i in range(n)]"),
        ("rms_norm", "matvec_prepared",
         "self._marshal_f32_output(out, n)",
         "[float(out[i]) for i in range(n)]"),
        ("dequantize_lookup_row", "report",
         "self._marshal_f32_output(out, n)",
         "[float(out[i]) for i in range(n)]"),
    )
    for case in cases:
        require_bulk(source, *case)

    print("QWEN38_BONSAI2_BULK_VECTOR_MARSHAL_WIRING_PASS")


if __name__ == "__main__":
    main()
