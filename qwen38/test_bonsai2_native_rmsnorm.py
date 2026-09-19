#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime
import qwen35_k3_full64_ggml_exact as exact
import qwen35_k3_full64_ggml_rmsnorm as rmswrap


def metadata() -> dict[str, object]:
    return {
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 128,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "identity",
        "prism.hadamard.weight_names": [],
        "prism.hadamard.inverse_weight_names": [],
        "prism.hadamard.gdn_v_grouped": False,
    }


def f32_bits(x: float) -> bytes:
    return struct.pack("<f", float(x))


def values(n: int, seed: int, scale: float) -> list[float]:
    state = seed & 0xFFFFFFFF
    out: list[float] = []
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        raw = ((state >> 8) % 40001 - 20000) / 20000.0
        out.append(exact.f32(raw * scale))
    return out


def reference(values_: list[float], rows: int, weight: list[float]) -> list[float]:
    width = len(weight)
    out: list[float] = []
    for row in range(rows):
        start = row * width
        out.extend(rmswrap.ggml_rms_norm(values_[start:start+width], weight, 1e-6))
    return out


def check(rt: Bonsai2NativeRuntime, rows: int, width: int, seed: int) -> None:
    x = values(rows * width, seed, 3.0)
    w = values(width, seed ^ 0xA5A5A5A5, 1.25)
    want = reference(x, rows, w)
    got = rt.rms_norm(x, w, rows=rows, eps=1e-6)
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(want, got)):
        if f32_bits(a) != f32_bits(b):
            raise AssertionError(
                f"RMSNorm mismatch rows={rows} width={width} index={i} "
                f"ref={a!r} got={b!r} "
                f"ref_bits={f32_bits(a).hex()} got_bits={f32_bits(b).hex()}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-lib", type=Path, required=True)
    args = ap.parse_args()

    exact.install()
    rt = Bonsai2NativeRuntime(args.native_lib, metadata(), threads=1, max_rows=4)
    try:
        assert hasattr(rt, "rms_norm"), "missing native exact-F32 RMSNorm API"
        check(rt, 1, 5120, 0x13555120)
        check(rt, 24, 256, 0x24256256)
        check(rt, 4, 256, 0x04256256)
        check(rt, 48, 128, 0x48128128)
    finally:
        rt.close()

    print("QWEN38_BONSAI2_NATIVE_RMSNORM_BITWISE_PASS")


if __name__ == "__main__":
    main()
