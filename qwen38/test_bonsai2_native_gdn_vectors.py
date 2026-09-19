#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime
import bonsai2_two_token as t2
import qwen35_gdn_quant_layer_gate as gdn
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


def conv_reference(
    qkv: list[float],
    history: list[list[float]],
    kernels: list[float],
) -> list[float]:
    prior = history[-3:]
    out = [0.0] * gdn.CONV_DIM
    for c in range(gdn.CONV_DIM):
        cur = t2.mulf(qkv[c], kernels[c * gdn.CONV_KERNEL + 3])
        for lag, old in enumerate(reversed(prior), start=1):
            cur = t2.addf(
                cur,
                t2.mulf(old[c], kernels[c * gdn.CONV_KERNEL + 3 - lag]),
            )
        out[c] = t2.siluf(cur)
    return out


def norm_gate_reference(
    core: list[float],
    norm_w: list[float],
    z: list[float],
) -> list[float]:
    out: list[float] = []
    for h in range(gdn.V_HEADS):
        base = h * gdn.HEAD_DIM
        ch = core[base : base + gdn.HEAD_DIM]
        zh = z[base : base + gdn.HEAD_DIM]
        nh = rmswrap.ggml_rms_norm(ch, norm_w, gdn.RMS_EPS)
        out.extend(
            t2.mulf(nh[d], t2.siluf(zh[d]))
            for d in range(gdn.HEAD_DIM)
        )
    return out


def assert_bitwise(name: str, want: list[float], got: list[float]) -> None:
    assert len(got) == len(want), (name, len(want), len(got))
    for i, (a, b) in enumerate(zip(want, got)):
        if f32_bits(a) != f32_bits(b):
            raise AssertionError(
                f"{name} bit mismatch index={i} ref={a!r} got={b!r} "
                f"ref_bits={f32_bits(a).hex()} got_bits={f32_bits(b).hex()}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-lib", type=Path, required=True)
    args = ap.parse_args()

    exact.install()
    rt = Bonsai2NativeRuntime(args.native_lib, metadata(), threads=1, max_rows=4)
    try:
        missing = [
            name
            for name in ("gdn_conv_silu", "gdn_norm_gate")
            if not hasattr(rt, name)
        ]
        assert not missing, f"missing native recurrent vector API: {missing}"

        qkv = values(gdn.CONV_DIM, 0x13552727, 2.5)
        history = [
            values(gdn.CONV_DIM, 0x11111111, 1.75),
            values(gdn.CONV_DIM, 0x22222222, 1.50),
            values(gdn.CONV_DIM, 0x33333333, 1.25),
        ]
        kernels = values(
            gdn.CONV_DIM * gdn.CONV_KERNEL, 0xABCDEF01, 0.375
        )
        want_conv = conv_reference(qkv, history, kernels)
        got_conv = rt.gdn_conv_silu(qkv, history, kernels)
        assert_bitwise("gdn_conv_silu", want_conv, got_conv)

        core = values(gdn.VALUE_DIM, 0x24681357, 3.0)
        z = values(gdn.VALUE_DIM, 0xDEADBEEF, 4.0)
        norm_w = values(gdn.HEAD_DIM, 0x10203040, 1.25)
        want_gate = norm_gate_reference(core, norm_w, z)
        got_gate = rt.gdn_norm_gate(core, norm_w, z)
        assert_bitwise("gdn_norm_gate", want_gate, got_gate)
    finally:
        rt.close()

    print("QWEN38_BONSAI2_NATIVE_GDN_VECTORS_BITWISE_PASS")


if __name__ == "__main__":
    main()
