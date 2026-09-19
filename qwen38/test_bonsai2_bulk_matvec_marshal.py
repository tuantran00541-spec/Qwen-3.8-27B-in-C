#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime


def finite_from_bits(bits: int) -> float:
    value = struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("fixture must be finite")
    return value


def f32_bytes(values) -> bytes:
    return b"".join(struct.pack("<f", float(v)) for v in values)


def check_helper() -> None:
    assert hasattr(Bonsai2NativeRuntime, "_marshal_f32_output"), (
        "missing bulk F32 output marshal helper"
    )
    bits = (
        0x00000000,
        0x80000000,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x3EAAAAAB,
        0x3F800000,
        0xBF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
    )
    values = [finite_from_bits(b) for b in bits]
    values.extend(float(i - 211) / 37.0 for i in range(2048))
    out = (ctypes.c_float * len(values))()
    for i, value in enumerate(values):
        out[i] = value

    reference = [float(out[i]) for i in range(len(values))]
    candidate = Bonsai2NativeRuntime._marshal_f32_output(out, len(values))
    assert len(candidate) == len(reference)
    if f32_bytes(candidate) != f32_bytes(reference):
        raise AssertionError("bulk F32 output marshal is not bitwise exact")


def check_matvec_wiring() -> None:
    source = (ROOT / "qwen38" / "bonsai2_quant_runtime.py").read_text(
        encoding="utf-8"
    )
    start = source.index("    def matvec_prepared(")
    end = source.index("    def matvec(", start)
    body = source[start:end]
    calls = body.count("self._marshal_f32_output(out, rows)")
    if calls != 2:
        raise AssertionError(
            f"matvec_prepared bulk-marshal calls={calls} expected=2"
        )
    if "[float(out[i]) for i in range(rows)]" in body:
        raise AssertionError("scalar Python matvec output copy still wired")


def main() -> None:
    check_helper()
    check_matvec_wiring()
    print("QWEN38_BONSAI2_BULK_MATVEC_MARSHAL_BITWISE_PASS")


if __name__ == "__main__":
    main()
