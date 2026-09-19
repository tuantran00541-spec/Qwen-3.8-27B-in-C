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


def reference(pregate: list[float], gate: list[float]) -> list[float]:
    return [
        exact.mul_f32(pregate[i], exact.sigmoid_f32(gate[i]))
        for i in range(len(pregate))
    ]


def check(rt: Bonsai2NativeRuntime, n: int, seed: int) -> None:
    pregate = values(n, seed, 7.0)
    gate = values(n, seed ^ 0xA5A5A5A5, 18.0)
    if n >= 10:
        gate[:10] = [
            exact.f32(v)
            for v in (-30.0, -20.0, -5.0, -1.0, -0.0, 0.0, 1.0, 5.0, 20.0, 30.0)
        ]
    want = reference(pregate, gate)
    got = rt.attention_sigmoid_mul(pregate, gate)
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(want, got)):
        if f32_bits(a) != f32_bits(b):
            raise AssertionError(
                f"attention gate mismatch n={n} index={i} "
                f"ref={a!r} got={b!r} "
                f"ref_bits={f32_bits(a).hex()} got_bits={f32_bits(b).hex()}"
            )


def check_execution_wiring() -> None:
    targets = {
        "bonsai2_full64_one_token.py": 1,
        "bonsai2_two_token.py": 1,
        "bonsai2_prompt_spike.py": 1,
    }
    forbidden = (
        "gate_sigmoid = [",
        "gs = [",
    )
    for filename, expected_calls in targets.items():
        source = (ROOT / "qwen38" / filename).read_text(encoding="utf-8")
        native_calls = source.count("runtime.attention_sigmoid_mul(")
        if native_calls != expected_calls:
            raise AssertionError(
                f"{filename}: native attention gate calls={native_calls} "
                f"expected={expected_calls}"
            )
        leftovers = [needle for needle in forbidden if needle in source]
        if leftovers:
            raise AssertionError(
                f"{filename}: Python attention gate still wired: {leftovers}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-lib", type=Path, required=True)
    args = ap.parse_args()

    exact.install()
    rt = Bonsai2NativeRuntime(args.native_lib, metadata(), threads=1, max_rows=4)
    try:
        assert hasattr(rt, "attention_sigmoid_mul"), (
            "missing fused native attention sigmoid-multiply API"
        )
        for n, seed in (
            (1, 0x11),
            (17, 0x1717),
            (128, 0x128128),
            (6144, 0x61446144),
        ):
            check(rt, n, seed)
        report = rt.report()
        assert report["timing_calls"]["attention_gate"] == 4
        check_execution_wiring()
    finally:
        rt.close()

    print("QWEN38_BONSAI2_NATIVE_ATTENTION_GATE_BITWISE_PASS")


if __name__ == "__main__":
    main()
