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


def values(n: int, seed: int, scale: float) -> list[float]:
    state = seed & 0xFFFFFFFF
    out: list[float] = []
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        raw = ((state >> 8) % 40001 - 20000) / 20000.0
        out.append(exact.f32(raw * scale))
    return out


def addf(a: float, b: float) -> float:
    return exact.f32(exact.f32(a) + exact.f32(b))


def check(rt: Bonsai2NativeRuntime, n: int, seed: int) -> None:
    a = values(n, seed, 11.0)
    b = values(n, seed ^ 0x5A5A5A5A, 13.0)
    want = [addf(x, y) for x, y in zip(a, b)]
    got = rt.residual_add(a, b)
    assert len(got) == len(want)
    for i, (x, y) in enumerate(zip(want, got)):
        if f32_bits(x) != f32_bits(y):
            raise AssertionError(
                f"residual add mismatch n={n} index={i} "
                f"ref={x!r} got={y!r} "
                f"ref_bits={f32_bits(x).hex()} got_bits={f32_bits(y).hex()}"
            )


def check_execution_wiring() -> None:
    targets = {
        "bonsai2_full64_one_token.py": 4,
        "bonsai2_two_token.py": 4,
        "bonsai2_prompt_spike.py": 4,
    }
    forbidden = (
        "residual = [addf(",
        "return [addf(residual",
        "final = [addf(residual",
        "residual = [float(hidden[i]) +",
        "return [residual[i] +",
    )
    for filename, expected_calls in targets.items():
        source = (ROOT / "qwen38" / filename).read_text(encoding="utf-8")
        native_calls = source.count("runtime.residual_add(")
        if native_calls != expected_calls:
            raise AssertionError(
                f"{filename}: native residual-add calls={native_calls} "
                f"expected={expected_calls}"
            )
        leftovers = [needle for needle in forbidden if needle in source]
        if leftovers:
            raise AssertionError(
                f"{filename}: Python residual add still wired: {leftovers}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-lib", type=Path, required=True)
    args = ap.parse_args()

    exact.install()
    rt = Bonsai2NativeRuntime(args.native_lib, metadata(), threads=1, max_rows=4)
    try:
        assert hasattr(rt, "residual_add"), "missing exact native residual-add API"
        for n, seed in (
            (1, 0x1),
            (17, 0x1717),
            (128, 0x128128),
            (5120, 0x51205120),
        ):
            check(rt, n, seed)
        report = rt.report()
        assert report["timing_calls"]["residual_add"] == 4
        check_execution_wiring()
    finally:
        rt.close()

    print("QWEN38_BONSAI2_NATIVE_RESIDUAL_ADD_BITWISE_PASS")


if __name__ == "__main__":
    main()
