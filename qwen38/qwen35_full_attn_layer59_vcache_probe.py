#!/usr/bin/env python3
"""Diagnose the remaining layer-59 local spike at the V -> F16 KV-cache boundary.

The combined full64 evidence already proves pinned-GGML RMSNorm and F32
attention gate arithmetic.  Layer 59 still shows a tiny V projection error
(~2e-7 relative) that can cross an IEEE-F16 rounding boundary and become a
larger one-key attention-value error.  This probe injects the oracle Vcur or
oracle attn_pregate to distinguish cache quantization sensitivity from an
attention/FFN semantic error.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as full64
import qwen35_k3_full64_ggml_exact as exact
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

LAYER = 59
MODES = ("baseline", "inject_vcur", "inject_attn_pregate")


def metrics(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    return full64.metrics(ref, cand)


def _layer_meta(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == LAYER)
    return {t["name"]: t for t in entry["tensors"]}


def run_layer(runtime, view, metas, vec, hidden: Sequence[float], ref: dict[str, list[float]], *, mode: str):
    if mode not in MODES:
        raise ValueError(mode)
    prefix = f"blk.{LAYER}"
    cp: dict[str, list[float]] = {}

    attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    cp["attn_norm-59"] = attn_norm

    qg = runtime.matvec(view("attn_q.weight"), metas[f"{prefix}.attn_q.weight"], attn_norm)
    _q, gate = attn.split_q_gate(qg)
    v = runtime.matvec(view("attn_v.weight"), metas[f"{prefix}.attn_v.weight"], attn_norm)
    cp["gate_reshaped-59"] = gate
    cp["Vcur_native-59"] = list(v)

    if mode == "inject_vcur":
        v = list(map(float, ref["Vcur-59"]))
    cp["Vcur_used-59"] = list(v)

    pregate = attn.gqa_one_key_attention(attn.f16_roundtrip(v))
    if mode == "inject_attn_pregate":
        pregate = list(map(float, ref["attn_pregate-59"]))
    cp["attn_pregate-59"] = pregate

    gate_sigmoid = [exact.sigmoid_f32(x) for x in gate]
    gated = [exact.mul_f32(pregate[i], gate_sigmoid[i]) for i in range(attn.Q_DIM)]
    cp["gate_sigmoid-59"] = gate_sigmoid
    cp["attn_gated-59"] = gated

    attn_out = runtime.matvec(view("attn_output.weight"), metas[f"{prefix}.attn_output.weight"], gated)
    cp["attn_output-59"] = attn_out
    residual = [float(hidden[i]) + attn_out[i] for i in range(gdn.HIDDEN)]
    cp["attn_residual-59"] = residual

    post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    cp["attn_post_norm-59"] = post_norm
    ffn_out = full64._run_ffn(runtime, view, metas, prefix, post_norm)
    cp["ffn_out-59"] = ffn_out
    cp["post_ffn-59"] = [residual[i] + ffn_out[i] for i in range(gdn.HIDDEN)]
    return cp


def execute(model: Path, native_lib: Path, oracle_json: Path, work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    exact.install()
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-layer59-ffn-oracle-v1" or not oracle.get("captured_complete"):
        raise RuntimeError("layer59 oracle incomplete")
    ref = oracle["checkpoints"]
    hidden = list(map(float, ref["layer59_input"]))

    directory = parse_gguf(model)
    trunk = work_dir / "layer59.k3.bin"
    manifest_path = work_dir / "layer59.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, manifest_path, layers=[LAYER], model_id=gdn.MODEL_ID,
        revision=gdn.REVISION, source_sha256=gdn.SHA256, expected_layers=gdn.DECODER_LAYERS,
    )
    runtime = gdn.QuantRuntime(gdn._load_native(native_lib))
    budget = int(manifest["layers"][0]["read_bytes"])

    with K3Trunk(trunk, manifest_path, budget_bytes=budget, want_ring=1,
                 max_pinned=0, prefer_direct_io=True) as reader:
        bound = reader.bind(LAYER)
        metas = _layer_meta(manifest)
        def view(s: str): return reader.tensor_view(bound, f"blk.{LAYER}.{s}")
        def vec(s: str): return gdn.f32_vector(view(s))

        ordered = [
            "attn_norm-59", "gate_reshaped-59", "attn_pregate-59",
            "gate_sigmoid-59", "attn_gated-59", "attn_output-59",
            "attn_residual-59", "attn_post_norm-59", "ffn_out-59", "post_ffn-59",
        ]
        modes: dict[str, Any] = {}
        for mode in MODES:
            cp = run_layer(runtime, view, metas, vec, hidden, ref, mode=mode)
            comp = {k: metrics(ref[k], cp[k]) for k in ordered if k in ref and k in cp}
            comp["Vcur_native-59"] = metrics(ref["Vcur-59"], cp["Vcur_native-59"])
            comp["Vcur_used-59"] = metrics(ref["Vcur-59"], cp["Vcur_used-59"])
            modes[mode] = {"checkpoint_metrics": comp, "output_metrics": comp["post_ffn-59"]}
        report = reader.report()
        bound.release()

    out_rel = {m: modes[m]["output_metrics"]["relative_l2"] for m in MODES}
    out_max = {m: modes[m]["output_metrics"]["max_abs"] for m in MODES}
    result = {
        "schema": "qwen38-full-attn-layer59-vcache-probe-v1",
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
        "cache_contract": "Vcur F32 -> default llama.cpp F16 KV cache -> one-key GQA value",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def sanity() -> None:
    exact.install()
    probe = [0.333251953125, -0.333251953125, 1.00048828125]
    rt = attn.f16_roundtrip(probe)
    if len(rt) != len(probe) or not all(math.isfinite(x) for x in rt) or len(MODES) != 3:
        raise SystemExit("layer59 V-cache sanity failed")
    print(json.dumps({
        "schema": "qwen38-full-attn-layer59-vcache-sanity-v1",
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
