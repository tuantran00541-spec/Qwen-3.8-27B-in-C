#!/usr/bin/env python3
"""Full64 evidence wrapper with the two proven pinned-GGML arithmetic fixes.

1. RMSNorm follows the pinned ggml F32 product / double accumulation / sqrtf
   semantics proven by the layer-50 microscope.
2. Full-attention sigmoid gating materializes F32 before the Q8_0 output
   projection, proven by the layer-35 pointwise microscope.

The base full64 executor remains unchanged until this combined wrapper passes a
real 64-layer artifact gate.
"""
from __future__ import annotations

import ctypes
import ctypes.util

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as full64
import qwen35_k3_full64_ggml_rmsnorm as rmswrap

_LIBM = None
_name = ctypes.util.find_library("m")
if _name:
    _LIBM = ctypes.CDLL(_name)
    _LIBM.expf.argtypes = [ctypes.c_float]
    _LIBM.expf.restype = ctypes.c_float


def f32(x: float) -> float:
    return rmswrap.f32(x)


def expf(x: float) -> float:
    if _LIBM is None:
        import math
        return f32(math.exp(f32(x)))
    return float(_LIBM.expf(ctypes.c_float(f32(x))))


def sigmoid_f32(x: float) -> float:
    # pinned ggml unary-ops.cpp: 1.f / (1.f + expf(-x))
    xf = f32(x)
    denom = f32(f32(1.0) + f32(expf(f32(-xf))))
    return f32(f32(1.0) / denom)


def mul_f32(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))


def run_full_attn_exact(runtime, view, metas, vec, hidden, layer):
    prefix = f"blk.{layer}"
    attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qg = runtime.matvec(view("attn_q.weight"), metas[f"{prefix}.attn_q.weight"], attn_norm)
    q, gate = attn.split_q_gate(qg)
    _q_norm = attn.rms_norm_heads(q, attn.N_HEAD, vec("attn_q_norm.weight"))
    k = runtime.matvec(view("attn_k.weight"), metas[f"{prefix}.attn_k.weight"], attn_norm)
    v = runtime.matvec(view("attn_v.weight"), metas[f"{prefix}.attn_v.weight"], attn_norm)
    k_norm = attn.rms_norm_heads(k, attn.N_HEAD_KV, vec("attn_k_norm.weight"))

    # Position 0: RoPE identity; cache K/V through default F16 storage.
    k_cache = attn.f16_roundtrip(k_norm)
    v_cache = attn.f16_roundtrip(v)
    pregate = attn.gqa_one_key_attention(v_cache)

    # Proven layer-35 fix: sigmoid is a F32 unary op and ggml_mul writes F32
    # before Q8_0 activation quantization for the output projection.
    gate_sigmoid = [sigmoid_f32(x) for x in gate]
    gated = [mul_f32(pregate[i], gate_sigmoid[i]) for i in range(attn.Q_DIM)]
    attn_out = runtime.matvec(
        view("attn_output.weight"), metas[f"{prefix}.attn_output.weight"], gated
    )
    residual = [float(hidden[i]) + attn_out[i] for i in range(gdn.HIDDEN)]
    post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    ffn_out = full64._run_ffn(runtime, view, metas, prefix, post_norm)
    final = [residual[i] + ffn_out[i] for i in range(gdn.HIDDEN)]
    return final, (len(k_cache) + len(v_cache)) * 2


def install() -> None:
    rmswrap.install()
    full64._run_full_attn_layer = run_full_attn_exact


def sanity() -> None:
    rmswrap.sanity()
    if sigmoid_f32(0.0) != 0.5:
        raise SystemExit("F32 sigmoid sanity failed")
    if mul_f32(0.25, 0.5) != 0.125:
        raise SystemExit("F32 gate multiply sanity failed")
    install()
    print("QWEN38_FULL64_GGML_EXACT_SANITY PASS")


if __name__ == "__main__":
    install()
    full64.main()
