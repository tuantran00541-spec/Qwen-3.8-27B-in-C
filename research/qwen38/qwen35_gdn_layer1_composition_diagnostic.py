#!/usr/bin/env python3
"""Layer-1 microscope for Qwen3.8 full-stack numerical composition.

This is intentionally diagnostic, not a relaxed correctness gate. It separates
(1) recurrent-layer semantic/layout errors, (2) amplification of tiny upstream
errors across quantized activation boundaries, and (3) missing F32
materialization inside or between ggml graph ops.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as full64
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

MODES = (
    "oracle_input_loose",
    "oracle_input_strict_f32",
    "candidate_input_loose",
    "candidate_boundary_f32_loose",
    "candidate_boundary_f32_strict",
    "candidate_layer0_strict_input_loose",
    "candidate_layer0_strict_input_strict",
)


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32v(xs: Sequence[float]) -> list[float]:
    return [f32(x) for x in xs]


def metrics(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    return full64.metrics(ref, cand)


def _layer_meta(manifest: dict[str, Any], layer: int) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == layer)
    return {t["name"]: t for t in entry["tensors"]}


def _maybe(xs: Sequence[float], strict: bool) -> list[float]:
    return f32v(xs) if strict else list(map(float, xs))


def run_recurrent(runtime: gdn.QuantRuntime, view, metas: dict[str, dict[str, Any]], vec,
                  hidden: Sequence[float], *, layer: int, strict: bool) -> dict[str, list[float]]:
    p = f"blk.{layer}"
    suffix = f"-{layer}"
    cp: dict[str, list[float]] = {}

    attn_norm = _maybe(gdn.rms_norm(hidden, vec("attn_norm.weight")), strict)
    cp[f"attn_norm{suffix}"] = attn_norm

    qkv = runtime.matvec(view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], attn_norm)
    z = runtime.matvec(view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], attn_norm)
    beta_raw = runtime.matvec(view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], attn_norm)
    alpha = runtime.matvec(view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], attn_norm)
    if strict:
        beta_raw, alpha = f32v(beta_raw), f32v(alpha)
    cp[f"linear_attn_qkv_mixed{suffix}"] = qkv
    cp[f"z{suffix}"] = z
    cp[f"beta{suffix}"] = beta_raw
    cp[f"alpha{suffix}"] = alpha

    beta = _maybe([gdn.sigmoid(x) for x in beta_raw], strict)
    cp[f"beta_sigmoid{suffix}"] = beta
    dt, a = vec("ssm_dt.bias"), vec("ssm_a")
    a_softplus = _maybe([gdn.softplus(alpha[i] + dt[i]) for i in range(gdn.V_HEADS)], strict)
    gate = _maybe([a[i] * a_softplus[i] for i in range(gdn.V_HEADS)], strict)
    cp[f"a_softplus{suffix}"] = a_softplus
    cp[f"gate{suffix}"] = gate

    kernels = vec("ssm_conv1d.weight")
    conv = _maybe([
        gdn.silu(qkv[c] * kernels[c * gdn.CONV_KERNEL + gdn.CONV_KERNEL - 1])
        for c in range(gdn.CONV_DIM)
    ], strict)
    cp[f"conv_output_silu{suffix}"] = conv
    q = conv[:gdn.KEY_DIM]
    k = conv[gdn.KEY_DIM:2 * gdn.KEY_DIM]
    v = conv[2 * gdn.KEY_DIM:]

    qn = [_maybe(gdn.l2_norm(h), strict) for h in gdn.split_heads(q, gdn.K_HEADS)]
    kn = [_maybe(gdn.l2_norm(h), strict) for h in gdn.split_heads(k, gdn.K_HEADS)]
    repeats = gdn.V_HEADS // gdn.K_HEADS
    cp[f"q_conv_predelta{suffix}"] = gdn.flatten([h for _ in range(repeats) for h in qn])
    cp[f"k_conv_predelta{suffix}"] = gdn.flatten([h for _ in range(repeats) for h in kn])
    cp[f"v_conv_predelta{suffix}"] = v

    core = _maybe(gdn.one_token_core(q, k, v, beta), strict)
    norm_w = vec("ssm_norm.weight")
    core_heads = gdn.split_heads(core, gdn.V_HEADS)
    z_heads = gdn.split_heads(z, gdn.V_HEADS)
    gated: list[list[float]] = []
    for ch, zh in zip(core_heads, z_heads):
        inv = 1.0 / math.sqrt(math.fsum(x * x for x in ch) / gdn.HEAD_DIM + gdn.RMS_EPS)
        if strict:
            normed = f32v([ch[d] * inv * norm_w[d] for d in range(gdn.HEAD_DIM)])
            siluz = f32v([gdn.silu(zh[d]) for d in range(gdn.HEAD_DIM)])
            gated.append(f32v([normed[d] * siluz[d] for d in range(gdn.HEAD_DIM)]))
        else:
            gated.append([ch[d] * inv * norm_w[d] * gdn.silu(zh[d]) for d in range(gdn.HEAD_DIM)])

    linear_out = runtime.matvec(view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gdn.flatten(gated))
    cp[f"linear_attn_out{suffix}"] = linear_out
    residual = _maybe([float(hidden[i]) + linear_out[i] for i in range(gdn.HIDDEN)], strict)
    cp[f"attn_residual{suffix}"] = residual
    post_norm = _maybe(gdn.rms_norm(residual, vec("post_attention_norm.weight")), strict)
    cp[f"attn_post_norm{suffix}"] = post_norm

    prepared = runtime.quantize(post_norm, "Q6_K")
    fgate = runtime.matvec(view("ffn_gate.weight"), metas[f"{p}.ffn_gate.weight"], post_norm, prepared)
    fup = runtime.matvec(view("ffn_up.weight"), metas[f"{p}.ffn_up.weight"], post_norm, prepared)
    swiglu = _maybe([gdn.silu(fgate[i]) * fup[i] for i in range(gdn.INTERMEDIATE)], strict)
    ffn_out = runtime.matvec(view("ffn_down.weight"), metas[f"{p}.ffn_down.weight"], swiglu)
    cp[f"ffn_out{suffix}"] = ffn_out
    cp[f"post_ffn{suffix}"] = _maybe([residual[i] + ffn_out[i] for i in range(gdn.HIDDEN)], strict)
    return cp


def execute(model: Path, native_lib: Path, oracle_json: Path, work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-layer1-composition-oracle-v1" or not oracle.get("captured_complete"):
        raise RuntimeError("layer1 composition oracle incomplete")
    ref = oracle["checkpoints"]
    oracle0 = list(map(float, ref["post_ffn-0"]))

    directory = parse_gguf(model)
    trunk, manifest_path = work_dir / "layers01.k3.bin", work_dir / "layers01.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, manifest_path, layers=[0, 1], model_id=gdn.MODEL_ID,
        revision=gdn.REVISION, source_sha256=gdn.SHA256, expected_layers=gdn.DECODER_LAYERS,
    )
    budget = max(int(x["read_bytes"]) for x in manifest["layers"])
    runtime = gdn.QuantRuntime(gdn._load_native(native_lib))
    embed = gdn._embedding_row(model, directory, int(oracle["token_id"]))

    with K3Trunk(trunk, manifest_path, budget_bytes=budget, want_ring=1, max_pinned=0, prefer_direct_io=True) as reader:
        b0 = reader.bind(0)
        m0 = _layer_meta(manifest, 0)
        def v0(s: str): return reader.tensor_view(b0, f"blk.0.{s}")
        def x0(s: str): return gdn.f32_vector(v0(s))
        candidate0 = run_recurrent(runtime, v0, m0, x0, embed, layer=0, strict=False)["post_ffn-0"]
        candidate0_strict = run_recurrent(runtime, v0, m0, x0, embed, layer=0, strict=True)["post_ffn-0"]
        b0.release()

        b1 = reader.bind(1)
        m1 = _layer_meta(manifest, 1)
        def v1(s: str): return reader.tensor_view(b1, f"blk.1.{s}")
        def x1(s: str): return gdn.f32_vector(v1(s))

        inputs = {
            "oracle_input_loose": (oracle0, False),
            "oracle_input_strict_f32": (oracle0, True),
            "candidate_input_loose": (candidate0, False),
            "candidate_boundary_f32_loose": (f32v(candidate0), False),
            "candidate_boundary_f32_strict": (f32v(candidate0), True),
            "candidate_layer0_strict_input_loose": (candidate0_strict, False),
            "candidate_layer0_strict_input_strict": (candidate0_strict, True),
        }
        modes: dict[str, Any] = {}
        for name, (inp, strict) in inputs.items():
            cp = run_recurrent(runtime, v1, m1, x1, inp, layer=1, strict=strict)
            comparisons = {k: metrics(ref[k], v) for k, v in cp.items() if k in ref}
            modes[name] = {
                "strict_f32_ops": strict,
                "input_metrics_vs_oracle_layer0": metrics(oracle0, inp),
                "checkpoint_metrics": comparisons,
                "output_metrics": comparisons["post_ffn-1"],
            }
        report = reader.report()
        b1.release()

    out_rel = {k: v["output_metrics"]["relative_l2"] for k, v in modes.items()}
    best = min(out_rel, key=out_rel.get)
    result = {
        "schema": "qwen38-gdn-layer1-composition-diagnostic-v2",
        "status": "PASS",
        "token_id": int(oracle["token_id"]),
        "candidate_layer0_metrics": metrics(oracle0, candidate0),
        "candidate_layer0_f32_boundary_metrics": metrics(oracle0, f32v(candidate0)),
        "candidate_layer0_strict_metrics": metrics(oracle0, candidate0_strict),
        "modes": modes,
        "best_mode": best,
        "best_output_relative_l2": out_rel[best],
        "reader_report": report,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def sanity() -> None:
    xs = [1.0 + 2.0**-25, -1.0 - 2.0**-25, 1e-40]
    ys = f32v(xs)
    if ys[0] != 1.0 or ys[1] != -1.0 or len(ys) != len(xs):
        raise SystemExit("F32 materialization sanity failed")
    if len(MODES) != 7 or len(set(MODES)) != 7:
        raise SystemExit("mode contract failed")
    print(json.dumps({"schema":"qwen38-gdn-layer1-composition-sanity-v2","status":"PASS","modes":MODES}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--oracle", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "sanity":
        sanity()
    else:
        a.work_dir.mkdir(parents=True, exist_ok=True)
        execute(a.model, a.native_lib, a.oracle, a.work_dir, a.output)


if __name__ == "__main__":
    main()
