#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime
import qwen35_full_attn_layer3_gate as attn
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


def f32(x: float) -> float:
    return exact.f32(x)


def addf(a: float, b: float) -> float:
    return f32(f32(a) + f32(b))


def mulf(a: float, b: float) -> float:
    return exact.mul_f32(a, b)


def f32_bytes(values) -> bytes:
    return b"".join(struct.pack("<f", float(x)) for x in values)


def values(n: int, seed: int, scale: float) -> list[float]:
    state = seed & 0xFFFFFFFF
    out: list[float] = []
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        raw = ((state >> 8) % 40001 - 20000) / 20000.0
        out.append(f32(raw * scale))
    return out


def reference(
    q: list[float],
    cache: dict[str, list[list[float]]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
) -> list[float]:
    if q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    repeat = q_heads // kv_heads
    n_ctx = len(cache["k"])
    out: list[float] = []
    for qidx in range(q_heads):
        kvh = qidx // repeat
        qv = q[qidx * head_dim:(qidx + 1) * head_dim]
        scores: list[float] = []
        for ti in range(n_ctx):
            kh = cache["k"][ti][kvh * head_dim:(kvh + 1) * head_dim]
            scores.append(f32(
                math.fsum(float(qv[d]) * float(kh[d]) for d in range(head_dim))
                * scale
            ))
        m = max(scores)
        exps = [f32(exact.expf(f32(s - m))) for s in scores]
        denom = f32(0.0)
        for value in exps:
            denom = addf(denom, value)
        probs = [f32(value / denom) for value in exps]
        for d in range(head_dim):
            acc = f32(0.0)
            for ti in range(n_ctx):
                vv = cache["v"][ti][kvh * head_dim + d]
                acc = addf(acc, mulf(probs[ti], vv))
            out.append(acc)
    return out


def run_case(
    rt: Bonsai2NativeRuntime,
    *,
    layer: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    n_ctx: int,
    seed: int,
) -> None:
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    scale = 1.0 / math.sqrt(head_dim)
    cache = {"k": [], "v": []}

    for ti in range(n_ctx):
        cache["k"].append(values(kv_dim, seed ^ (ti * 0x13579 + 0x1111), 3.0))
        cache["v"].append(values(kv_dim, seed ^ (ti * 0x2468B + 0x2222), 2.0))
        q = values(q_dim, seed ^ (ti * 0x31415 + 0x3333), 4.0)
        want = reference(q, cache, q_heads, kv_heads, head_dim, scale)
        got = rt.attention_core(
            layer,
            q,
            cache,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=scale,
        )
        if f32_bytes(want) != f32_bytes(got):
            for i, (a, b) in enumerate(zip(want, got)):
                if struct.pack("<f", a) != struct.pack("<f", b):
                    raise AssertionError(
                        f"attention core mismatch layer={layer} ctx={ti + 1} "
                        f"index={i} ref={a!r} got={b!r}"
                    )
            raise AssertionError("attention core length mismatch")


def check_execution_wiring() -> None:
    expected = {
        "bonsai2_full64_one_token.py": (
            1,
            ("pregate = attn.gqa_one_key_attention(v_cache)",),
        ),
        "bonsai2_two_token.py": (
            1,
            ("qh = attn.split_heads(q_rope, attn.N_HEAD)",),
        ),
        "bonsai2_prompt_spike.py": (
            1,
            ("scores: list[float] = []", "probs = softmax_many(scores)"),
        ),
    }
    for filename, (expected_calls, forbidden) in expected.items():
        source = (ROOT / "qwen38" / filename).read_text(encoding="utf-8")
        calls = source.count("runtime.attention_core(")
        if calls != expected_calls:
            raise AssertionError(
                f"{filename}: native attention-core calls={calls} expected={expected_calls}"
            )
        leftovers = [needle for needle in forbidden if needle in source]
        if leftovers:
            raise AssertionError(
                f"{filename}: Python attention core still wired: {leftovers}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-lib", type=Path, required=True)
    args = ap.parse_args()

    exact.install()
    rt = Bonsai2NativeRuntime(args.native_lib, metadata(), threads=1, max_rows=4)
    try:
        assert hasattr(rt, "attention_core"), "missing exact native attention-core API"
        run_case(
            rt, layer=7, q_heads=6, kv_heads=2, head_dim=32,
            n_ctx=7, seed=0x7032,
        )
        run_case(
            rt, layer=11, q_heads=attn.N_HEAD, kv_heads=attn.N_HEAD_KV,
            head_dim=attn.HEAD_DIM, n_ctx=2, seed=0xB024,
        )
        report = rt.report()
        assert report["timing_calls"]["attention_core"] == 9
        check_execution_wiring()
    finally:
        rt.close()

    print("QWEN38_BONSAI2_NATIVE_ATTENTION_CORE_BITWISE_PASS")


if __name__ == "__main__":
    main()
