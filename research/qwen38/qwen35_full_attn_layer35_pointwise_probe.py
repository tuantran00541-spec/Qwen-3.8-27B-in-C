#!/usr/bin/env python3
"""Diagnose the isolated layer-35 full-attention local spike.

The full64 GGML-RMSNorm run removed the recurrent layer-50 cliff but exposed a
single hard local failure at full-attention layer 35.  This probe keeps the
proven RMSNorm fix and tests whether llama.cpp's F32 sigmoid + multiply
materialization before the Q8_0 attention-output projection removes the spike.
Oracle injection modes separate that attention pointwise boundary from the
post-attention FFN.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as full64
import qwen35_k3_full64_ggml_rmsnorm as rmswrap
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

LAYER = 35
PREV = 34
MODES = ("baseline", "f32_gate_mul", "inject_attn_gated", "inject_attn_output")

_LIBM = None
try:
    _libm_name = ctypes.util.find_library("m")
    if _libm_name:
        _LIBM = ctypes.CDLL(_libm_name)
        _LIBM.expf.argtypes = [ctypes.c_float]
        _LIBM.expf.restype = ctypes.c_float
except Exception:
    _LIBM = None


def f32(x: float) -> float:
    return rmswrap.f32(x)


def expf(x: float) -> float:
    xf = f32(x)
    if _LIBM is not None:
        return float(_LIBM.expf(ctypes.c_float(xf)))
    return f32(math.exp(xf))


def sigmoid_f32(x: float) -> float:
    # pinned ggml unary-ops.cpp: 1.f / (1.f + expf(-x))
    xf = f32(x)
    e = expf(f32(-xf))
    denom = f32(f32(1.0) + f32(e))
    return f32(f32(1.0) / denom)


def mul_f32(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))


def metrics(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    return full64.metrics(ref, cand)


def _layer_meta(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == LAYER)
    return {t["name"]: t for t in entry["tensors"]}


def run_layer35(runtime: gdn.QuantRuntime, view, metas: dict[str, dict[str, Any]], vec,
                hidden: Sequence[float], ref: dict[str, list[float]], *, mode: str) -> dict[str, list[float]]:
    if mode not in MODES:
        raise ValueError(mode)
    prefix = f"blk.{LAYER}"
    cp: dict[str, list[float]] = {}

    attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    cp["attn_norm-35"] = attn_norm

    qg = runtime.matvec(view("attn_q.weight"), metas[f"{prefix}.attn_q.weight"], attn_norm)
    _q, gate = attn.split_q_gate(qg)
    v = runtime.matvec(view("attn_v.weight"), metas[f"{prefix}.attn_v.weight"], attn_norm)
    cp["Qcur_full-35"] = qg
    cp["gate_reshaped-35"] = gate
    cp["Vcur-35"] = v

    # Position 0 has one cached key, so softmax is exactly 1.  Attention value
    # is V after the default F16 cache round-trip and grouped KV expansion.
    pregate = attn.gqa_one_key_attention(attn.f16_roundtrip(v))
    cp["attn_pregate-35"] = pregate

    if mode == "f32_gate_mul":
        gate_sigmoid = [sigmoid_f32(x) for x in gate]
        gated = [mul_f32(pregate[i], gate_sigmoid[i]) for i in range(attn.Q_DIM)]
    else:
        gate_sigmoid = [gdn.sigmoid(x) for x in gate]
        gated = [pregate[i] * gate_sigmoid[i] for i in range(attn.Q_DIM)]
    cp["gate_sigmoid-35"] = gate_sigmoid

    if mode == "inject_attn_gated":
        gated = list(map(float, ref["attn_gated-35"]))
    cp["attn_gated-35"] = gated

    attn_out = runtime.matvec(
        view("attn_output.weight"), metas[f"{prefix}.attn_output.weight"], gated
    )
    if mode == "inject_attn_output":
        attn_out = list(map(float, ref["attn_output-35"]))
    cp["attn_output-35"] = attn_out

    residual = [float(hidden[i]) + attn_out[i] for i in range(gdn.HIDDEN)]
    cp["attn_residual-35"] = residual
    post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    cp["attn_post_norm-35"] = post_norm
    ffn_out = full64._run_ffn(runtime, view, metas, prefix, post_norm)
    cp["ffn_out-35"] = ffn_out
    cp["post_ffn-35"] = [residual[i] + ffn_out[i] for i in range(gdn.HIDDEN)]
    return cp


def execute(model: Path, native_lib: Path, oracle_json: Path, work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    rmswrap.install()
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-layer35-pointwise-oracle-v1" or not oracle.get("captured_complete"):
        raise RuntimeError("layer35 pointwise oracle incomplete")
    ref = oracle["checkpoints"]
    hidden = list(map(float, ref["layer35_input"]))
    if len(hidden) != gdn.HIDDEN:
        raise ValueError("layer35 input width mismatch")

    directory = parse_gguf(model)
    trunk = work_dir / "layer35.k3.bin"
    manifest_path = work_dir / "layer35.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, manifest_path, layers=[LAYER], model_id=gdn.MODEL_ID,
        revision=gdn.REVISION, source_sha256=gdn.SHA256, expected_layers=gdn.DECODER_LAYERS,
    )
    budget = int(manifest["layers"][0]["read_bytes"])
    runtime = gdn.QuantRuntime(gdn._load_native(native_lib))

    with K3Trunk(trunk, manifest_path, budget_bytes=budget, want_ring=1,
                 max_pinned=0, prefer_direct_io=True) as reader:
        bound = reader.bind(LAYER)
        metas = _layer_meta(manifest)
        def view(s: str): return reader.tensor_view(bound, f"blk.{LAYER}.{s}")
        def vec(s: str): return gdn.f32_vector(view(s))

        modes: dict[str, Any] = {}
        ordered = [
            "attn_norm-35", "Qcur_full-35", "gate_reshaped-35", "Vcur-35",
            "attn_pregate-35", "gate_sigmoid-35", "attn_gated-35",
            "attn_output-35", "attn_residual-35", "attn_post_norm-35",
            "ffn_out-35", "post_ffn-35",
        ]
        for mode in MODES:
            cp = run_layer35(runtime, view, metas, vec, hidden, ref, mode=mode)
            comparisons = {k: metrics(ref[k], cp[k]) for k in ordered if k in ref and k in cp}
            modes[mode] = {
                "checkpoint_metrics": comparisons,
                "output_metrics": comparisons["post_ffn-35"],
            }
        report = reader.report()
        bound.release()

    out_rel = {m: modes[m]["output_metrics"]["relative_l2"] for m in MODES}
    out_max = {m: modes[m]["output_metrics"]["max_abs"] for m in MODES}
    result = {
        "schema": "qwen38-full-attn-layer35-pointwise-probe-v1",
        "status": "PASS",
        "diagnostic_only": True,
        "token_id": int(oracle["token_id"]),
        "input_source": oracle.get("input_source"),
        "modes": modes,
        "output_relative_l2": out_rel,
        "output_max_abs": out_max,
        "best_mode_relative_l2": min(out_rel, key=out_rel.get),
        "best_mode_max_abs": min(out_max, key=out_max.get),
        "reader_report": report,
        "elapsed_seconds": time.monotonic() - started,
        "pointwise_contract": "sigmoid=1.f/(1.f+expf(-x)); ggml_mul materializes F32",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def sanity() -> None:
    rmswrap.install()
    p = sigmoid_f32(0.0)
    if p != 0.5 or mul_f32(0.25, p) != 0.125:
        raise SystemExit("F32 sigmoid/mul sanity failed")
    if LAYER % 4 != 3 or PREV != 34 or len(MODES) != 4:
        raise SystemExit("layer35 pointwise contract failed")
    print(json.dumps({
        "schema": "qwen38-full-attn-layer35-pointwise-sanity-v1",
        "status": "PASS", "layer": LAYER, "modes": MODES,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--oracle", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "sanity": sanity()
    else:
        a.work_dir.mkdir(parents=True, exist_ok=True)
        execute(a.model, a.native_lib, a.oracle, a.work_dir, a.output)


if __name__ == "__main__":
    main()
