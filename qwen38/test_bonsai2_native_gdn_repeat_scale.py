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


def f32_bytes(values) -> bytes:
    return b"".join(struct.pack("<f", float(v)) for v in values)


def values(n: int, seed: int, scale: float) -> list[float]:
    state = seed & 0xFFFFFFFF
    out: list[float] = []
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        raw = ((state >> 8) % 40001 - 20000) / 20000.0
        out.append(exact.f32(raw * scale))
    return out


def reference(
    q: list[float],
    k: list[float],
    *,
    repeats: int,
    scale: float,
) -> tuple[list[float], list[float]]:
    q_out: list[float] = []
    k_out: list[float] = []
    for _ in range(repeats):
        q_out.extend(exact.mul_f32(v, scale) for v in q)
        k_out.extend(k)
    return q_out, k_out


def check_case(
    rt: Bonsai2NativeRuntime,
    *,
    key_dim: int,
    repeats: int,
    seed: int,
) -> None:
    q = values(key_dim, seed, 3.0)
    k = values(key_dim, seed ^ 0x5A5A5A5A, 3.0)
    scale = exact.f32(1.0 / (128.0 ** 0.5))
    want_q, want_k = reference(q, k, repeats=repeats, scale=scale)
    got_q, got_k = rt.gdn_repeat_scale(
        q, k, repeats=repeats, scale=scale
    )
    if f32_bytes(want_q) != f32_bytes(got_q):
        raise AssertionError(
            f"GDN q repeat-scale mismatch key_dim={key_dim} repeats={repeats}"
        )
    if f32_bytes(want_k) != f32_bytes(got_k):
        raise AssertionError(
            f"GDN k repeat mismatch key_dim={key_dim} repeats={repeats}"
        )


def check_execution_wiring() -> None:
    two_token = (ROOT / "qwen38" / "bonsai2_two_token.py").read_text(
        encoding="utf-8"
    )
    prompt = (ROOT / "qwen38" / "bonsai2_prompt_spike.py").read_text(
        encoding="utf-8"
    )

    if two_token.count("runtime.gdn_repeat_scale(") != 1:
        raise AssertionError(
            "bonsai2_two_token.py must keep one standalone exact repeat-scale call"
        )

    prompt_direct = prompt.count("runtime.gdn_repeat_scale(")
    prompt_fused = prompt.count("runtime.recurrent_mid(")
    if prompt_direct != 0 or prompt_fused != 1:
        raise AssertionError(
            "bonsai2_prompt_spike.py repeat-scale ownership must be fused into "
            f"one recurrent_mid call; direct={prompt_direct} fused={prompt_fused}"
        )

    forbidden = (
        "q48 = [mulf(vv, SCALE_GDN) for vv in repeat_k_heads(qn)]",
        "k48 = repeat_k_heads(kn)",
        "q48 = [mulf(value, t2.SCALE_GDN) for value in t2.repeat_k_heads(qn)]",
        "k48 = t2.repeat_k_heads(kn)",
    )
    leftovers = [
        needle for needle in forbidden
        if needle in two_token or needle in prompt
    ]
    if leftovers:
        raise AssertionError(
            f"Python GDN repeat-scale still wired: {leftovers}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-lib", type=Path, required=True)
    args = ap.parse_args()

    exact.install()
    rt = Bonsai2NativeRuntime(args.native_lib, metadata(), threads=1, max_rows=4)
    try:
        assert hasattr(rt, "gdn_repeat_scale"), (
            "missing exact native GDN repeat-scale API"
        )
        check_case(rt, key_dim=17, repeats=3, seed=0x173)
        check_case(rt, key_dim=2048, repeats=3, seed=0x20483)
        report = rt.report()
        assert report["timing_calls"]["gdn_repeat_scale"] == 2
        check_execution_wiring()
    finally:
        rt.close()

    print("QWEN38_BONSAI2_NATIVE_GDN_REPEAT_SCALE_BITWISE_PASS")


if __name__ == "__main__":
    main()
