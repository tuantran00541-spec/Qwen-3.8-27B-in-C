#!/usr/bin/env python3
"""Run the full64 K3 gate with pinned ggml F32 RMSNorm arithmetic.

This is an evidence wrapper: it monkeypatches the two Python RMSNorm helpers
without changing the proven base files.  The arithmetic contract comes from
llama.cpp pin 557614e0296ff4a5b6f649737a65ae2076eea2fd: F32 input squares are
cast to the double ggml_float accumulator, then mean/sqrtf/scale/output return
to F32.
"""
from __future__ import annotations

import math
import struct
from typing import Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as full64


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def ggml_rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = gdn.RMS_EPS) -> list[float]:
    if len(x) != len(weight):
        raise ValueError("RMSNorm shape mismatch")
    sum_sq = 0.0
    for value in x:
        vf = f32(value)
        sum_sq += float(f32(vf * vf))
    mean = f32(sum_sq / len(x))
    mean_eps = f32(mean + f32(eps))
    scale = f32(1.0 / f32(math.sqrt(mean_eps)))
    out: list[float] = []
    for value, w in zip(x, weight):
        out.append(f32(f32(f32(value) * scale) * f32(w)))
    return out


def ggml_rms_norm_heads(values: Sequence[float], heads: int, weight: Sequence[float]) -> list[float]:
    if len(weight) != attn.HEAD_DIM:
        raise ValueError("per-head RMSNorm weight width mismatch")
    split = attn.split_heads(values, heads)
    return attn.flatten([ggml_rms_norm(head, weight, attn.RMS_EPS) for head in split])


def install() -> None:
    gdn.rms_norm = ggml_rms_norm
    attn.rms_norm_heads = ggml_rms_norm_heads


def sanity() -> None:
    x = f32(1.0000001192092896)
    if f32(x * x) != 1.000000238418579:
        raise SystemExit("F32 product rounding sanity failed")
    probe = ggml_rms_norm([x, -x], [1.0, 1.0], 1e-6)
    if len(probe) != 2 or not all(math.isfinite(v) for v in probe):
        raise SystemExit("GGML RMSNorm wrapper sanity failed")
    heads = ggml_rms_norm_heads([0.25] * (2 * attn.HEAD_DIM), 2, [1.0] * attn.HEAD_DIM)
    if len(heads) != 2 * attn.HEAD_DIM:
        raise SystemExit("head RMSNorm wrapper sanity failed")
    print("QWEN38_FULL64_GGML_RMSNORM_SANITY PASS")


if __name__ == "__main__":
    install()
    # Preserve the base command line contract: `sanity` or `run --model ...`.
    full64.main()
