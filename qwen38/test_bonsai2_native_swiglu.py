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


def reference(gate: list[float], up: list[float]) -> list[float]:
    return [
        exact.mul_f32(
            exact.mul_f32(g, exact.sigmoid_f32(g)),
            u,
        )
        for g, u in zip(gate, up)
    ]


def values(n: int) -> tuple[list[float], list[float]]:
    gate = [-30.0, -20.0, -4.0, -1.0, -0.0, 0.0, 0.5, 1.0, 4.0, 20.0, 30.0]
    up = [2.0, -1.0, 0.25, -0.5, 3.0, -3.0, 1.5, -2.0, 0.125, 8.0, -8.0]
    state = 0x13552727
    while len(gate) < n:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        a = ((state >> 8) % 40001 - 20000) / 2048.0
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        b = ((state >> 8) % 40001 - 20000) / 4096.0
        gate.append(exact.f32(a))
        up.append(exact.f32(b))
    return gate[:n], up[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-lib", type=Path, required=True)
    args = ap.parse_args()

    exact.install()
    rt = Bonsai2NativeRuntime(args.native_lib, metadata(), threads=1, max_rows=4)
    try:
        gate, up = values(17408)
        want = reference(gate, up)
        got = rt.swiglu(gate, up)
        assert len(got) == len(want)
        for i, (a, b) in enumerate(zip(want, got)):
            if f32_bits(a) != f32_bits(b):
                raise AssertionError(
                    f"SwiGLU bit mismatch index={i} ref={a!r} got={b!r} "
                    f"ref_bits={f32_bits(a).hex()} got_bits={f32_bits(b).hex()}"
                )
    finally:
        rt.close()
    print("QWEN38_BONSAI2_NATIVE_SWIGLU_BITWISE_PASS")


if __name__ == "__main__":
    main()
