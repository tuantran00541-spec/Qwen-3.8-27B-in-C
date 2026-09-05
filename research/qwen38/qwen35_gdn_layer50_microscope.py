#!/usr/bin/env python3
"""Layer-50 recurrent microscope for the Qwen3.8 GGUF K3 runtime.

This is diagnostic, not a relaxed correctness gate.  The full64 dual gate
found one isolated local-output spike at decoder layer 50 while neighboring
layers stayed close to the pinned llama.cpp oracle.  This microscope executes
only layer 50 from the exact oracle layer-49 boundary and adds controlled
oracle injections to locate the first sensitive stage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as full64
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

LAYER = 50
PREV = 49
MODES = ("baseline", "inject_attn_norm", "inject_projection_bundle")


def metrics(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    return full64.metrics(ref, cand)


def _layer_meta(manifest: dict[str, Any], layer: int) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == layer)
    return {t["name"]: t for t in entry["tensors"]}


def run_layer50(runtime: gdn.QuantRuntime, view, metas: dict[str, dict[str, Any]], vec,
                hidden: Sequence[float], ref: dict[str, list[float]], *, mode: str) -> dict[str, list[float]]:
    if mode not in MODES:
        raise ValueError(mode)
    p = "blk.50"
    cp: dict[str, list[float]] = {}

    computed_attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    attn_norm = list(map(float, ref["attn_norm-50"])) if mode == "inject_attn_norm" else computed_attn_norm
    cp["attn_norm-50"] = attn_norm

    if mode == "inject_projection_bundle":
        qkv = list(map(float, ref["linear_attn_qkv_mixed-50"]))
        z = list(map(float, ref["z-50"]))
        beta_raw = list(map(float, ref["beta-50"]))
        alpha = list(map(float, ref["alpha-50"]))
    else:
        qkv = runtime.matvec(view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], attn_norm)
        z = runtime.matvec(view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], attn_norm)
        beta_raw = runtime.matvec(view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], attn_norm)
        alpha = runtime.matvec(view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], attn_norm)
    cp["linear_attn_qkv_mixed-50"] = qkv
    cp["z-50"] = z
    cp["beta-50"] = beta_raw
    cp["alpha-50"] = alpha

    beta = [gdn.sigmoid(x) for x in beta_raw]
    cp["beta_sigmoid-50"] = beta
    dt, a = vec("ssm_dt.bias"), vec("ssm_a")
    a_softplus = [gdn.softplus(alpha[i] + dt[i]) for i in range(gdn.V_HEADS)]
    gate = [a[i] * a_softplus[i] for i in range(gdn.V_HEADS)]
    cp["a_softplus-50"] = a_softplus
    cp["gate-50"] = gate

    kernels = vec("ssm_conv1d.weight")
    conv = [
        gdn.silu(qkv[c] * kernels[c * gdn.CONV_KERNEL + gdn.CONV_KERNEL - 1])
        for c in range(gdn.CONV_DIM)
    ]
    cp["conv_output_silu-50"] = conv
    q = conv[:gdn.KEY_DIM]
    k = conv[gdn.KEY_DIM:2 * gdn.KEY_DIM]
    v = conv[2 * gdn.KEY_DIM:]

    qn = [gdn.l2_norm(h) for h in gdn.split_heads(q, gdn.K_HEADS)]
    kn = [gdn.l2_norm(h) for h in gdn.split_heads(k, gdn.K_HEADS)]
    repeats = gdn.V_HEADS // gdn.K_HEADS
    cp["q_conv_predelta-50"] = gdn.flatten([h for _ in range(repeats) for h in qn])
    cp["k_conv_predelta-50"] = gdn.flatten([h for _ in range(repeats) for h in kn])
    cp["v_conv_predelta-50"] = v

    core = gdn.one_token_core(q, k, v, beta)
    norm_w = vec("ssm_norm.weight")
    core_heads = gdn.split_heads(core, gdn.V_HEADS)
    z_heads = gdn.split_heads(z, gdn.V_HEADS)
    gated: list[list[float]] = []
    for ch, zh in zip(core_heads, z_heads):
        inv = 1.0 / math.sqrt(math.fsum(x*x for x in ch) / gdn.HEAD_DIM + gdn.RMS_EPS)
        gated.append([
            ch[d] * inv * norm_w[d] * gdn.silu(zh[d])
            for d in range(gdn.HEAD_DIM)
        ])

    linear_out = runtime.matvec(view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gdn.flatten(gated))
    cp["linear_attn_out-50"] = linear_out
    residual = [float(hidden[i]) + linear_out[i] for i in range(gdn.HIDDEN)]
    cp["attn_residual-50"] = residual
    post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    cp["attn_post_norm-50"] = post_norm

    prepared = runtime.quantize(post_norm, "Q6_K")
    fgate = runtime.matvec(view("ffn_gate.weight"), metas[f"{p}.ffn_gate.weight"], post_norm, prepared)
    fup = runtime.matvec(view("ffn_up.weight"), metas[f"{p}.ffn_up.weight"], post_norm, prepared)
    swiglu = [gdn.silu(fgate[i]) * fup[i] for i in range(gdn.INTERMEDIATE)]
    ffn_out = runtime.matvec(view("ffn_down.weight"), metas[f"{p}.ffn_down.weight"], swiglu)
    cp["ffn_out-50"] = ffn_out
    cp["post_ffn-50"] = [residual[i] + ffn_out[i] for i in range(gdn.HIDDEN)]
    return cp


def execute(model: Path, native_lib: Path, oracle_json: Path, work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-layer50-microscope-oracle-v1" or not oracle.get("captured_complete"):
        raise RuntimeError("layer50 microscope oracle incomplete")
    ref = oracle["checkpoints"]
    hidden = list(map(float, ref["post_ffn-49"]))

    directory = parse_gguf(model)
    trunk, manifest_path = work_dir / "layer50.k3.bin", work_dir / "layer50.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, manifest_path, layers=[LAYER], model_id=gdn.MODEL_ID,
        revision=gdn.REVISION, source_sha256=gdn.SHA256, expected_layers=gdn.DECODER_LAYERS,
    )
    budget = int(manifest["layers"][0]["read_bytes"])
    runtime = gdn.QuantRuntime(gdn._load_native(native_lib))

    with K3Trunk(trunk, manifest_path, budget_bytes=budget, want_ring=1, max_pinned=0, prefer_direct_io=True) as reader:
        bound = reader.bind(LAYER)
        metas = _layer_meta(manifest, LAYER)
        def view(s: str): return reader.tensor_view(bound, f"blk.{LAYER}.{s}")
        def vec(s: str): return gdn.f32_vector(view(s))

        modes: dict[str, Any] = {}
        for mode in MODES:
            cp = run_layer50(runtime, view, metas, vec, hidden, ref, mode=mode)
            comparisons = {k: metrics(ref[k], v) for k, v in cp.items() if k in ref}
            modes[mode] = {
                "checkpoint_metrics": comparisons,
                "output_metrics": comparisons["post_ffn-50"],
            }
        report = reader.report()
        bound.release()

    baseline = modes["baseline"]["checkpoint_metrics"]
    ordered = [
        "attn_norm-50", "linear_attn_qkv_mixed-50", "z-50", "beta-50", "alpha-50",
        "beta_sigmoid-50", "a_softplus-50", "gate-50", "conv_output_silu-50",
        "q_conv_predelta-50", "k_conv_predelta-50", "v_conv_predelta-50",
        "linear_attn_out-50", "attn_residual-50", "attn_post_norm-50", "ffn_out-50", "post_ffn-50",
    ]
    first_over_2e3 = next((k for k in ordered if k in baseline and baseline[k]["relative_l2"] > 2e-3), None)
    out_rel = {name: modes[name]["output_metrics"]["relative_l2"] for name in MODES}
    best = min(out_rel, key=out_rel.get)
    result = {
        "schema": "qwen38-gdn-layer50-microscope-v1",
        "status": "PASS",
        "diagnostic_only": True,
        "token_id": int(oracle["token_id"]),
        "input_source": "oracle post_ffn-49 derived FP32 residual+ffn",
        "modes": modes,
        "baseline_first_checkpoint_over_rel_2e3": first_over_2e3,
        "best_mode": best,
        "best_output_relative_l2": out_rel[best],
        "injection_effect": {
            "baseline_output_relative_l2": out_rel["baseline"],
            "inject_attn_norm_output_relative_l2": out_rel["inject_attn_norm"],
            "inject_projection_bundle_output_relative_l2": out_rel["inject_projection_bundle"],
        },
        "reader_report": report,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def sanity() -> None:
    if LAYER != 50 or PREV != 49 or len(MODES) != 3 or len(set(MODES)) != 3:
        raise SystemExit("layer50 microscope contract failed")
    if LAYER % 4 == 3:
        raise SystemExit("layer50 must be recurrent, not full attention")
    print(json.dumps({
        "schema": "qwen38-gdn-layer50-microscope-sanity-v1",
        "status": "PASS",
        "layer": LAYER,
        "previous_layer": PREV,
        "modes": MODES,
        "diagnostic_only": True,
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
