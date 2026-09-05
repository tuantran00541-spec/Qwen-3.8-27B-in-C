#!/usr/bin/env python3
"""Full64 one-token gate with explicit quantized-boundary semantics.

Free-running decoding always uses native K3 activations.  The local semantic
lane remains teacher-forced at decoder-layer boundaries; additionally, each
full-attention layer validates its native V projection against the already
proven layer-3 Vcur threshold, then injects the oracle F32 Vcur immediately
before the default F16 KV-cache store.  This separates a discontinuous F16
rounding cliff from downstream attention/FFN semantics without loosening any
post_ffn threshold or changing free-running behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as base
import qwen35_k3_full64_ggml_exact as exact
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

VCUR_LIMIT = (5e-4, 2e-4)  # exact threshold already gated by real layer-3 evidence
LOCAL_LAYER_LIMIT = base.LOCAL_LAYER_LIMIT
FREE_DRIFT_LAYER_LIMIT = base.FREE_DRIFT_LAYER_LIMIT


def run_full_attn(runtime: gdn.QuantRuntime, view, metas: dict[str, dict[str, Any]], vec,
                  hidden: Sequence[float], layer: int, *, v_override: Sequence[float] | None = None
                  ) -> tuple[list[float], int, list[float]]:
    prefix = f"blk.{layer}"
    attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qg = runtime.matvec(view("attn_q.weight"), metas[f"{prefix}.attn_q.weight"], attn_norm)
    q, gate = attn.split_q_gate(qg)
    _q_norm = attn.rms_norm_heads(q, attn.N_HEAD, vec("attn_q_norm.weight"))
    k = runtime.matvec(view("attn_k.weight"), metas[f"{prefix}.attn_k.weight"], attn_norm)
    v_native = runtime.matvec(view("attn_v.weight"), metas[f"{prefix}.attn_v.weight"], attn_norm)
    k_norm = attn.rms_norm_heads(k, attn.N_HEAD_KV, vec("attn_k_norm.weight"))

    v_used = list(map(float, v_override)) if v_override is not None else v_native
    if len(v_used) != attn.KV_DIM:
        raise ValueError(f"layer {layer}: Vcur override width={len(v_used)} expected={attn.KV_DIM}")

    k_cache = attn.f16_roundtrip(k_norm)
    v_cache = attn.f16_roundtrip(v_used)
    pregate = attn.gqa_one_key_attention(v_cache)
    gate_sigmoid = [exact.sigmoid_f32(x) for x in gate]
    gated = [exact.mul_f32(pregate[i], gate_sigmoid[i]) for i in range(attn.Q_DIM)]
    attn_out = runtime.matvec(
        view("attn_output.weight"), metas[f"{prefix}.attn_output.weight"], gated
    )
    residual = [float(hidden[i]) + attn_out[i] for i in range(gdn.HIDDEN)]
    post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    ffn_out = base._run_ffn(runtime, view, metas, prefix, post_norm)
    final = [residual[i] + ffn_out[i] for i in range(gdn.HIDDEN)]
    return final, (len(k_cache) + len(v_cache)) * 2, list(v_native)


def execute(model: Path, native_lib: Path, inventory_json: Path, oracle_json: Path,
            work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    exact.install()

    inv = json.loads(inventory_json.read_text(encoding="utf-8"))
    if inv.get("status") != "PASS" or inv.get("sha256") != gdn.SHA256:
        raise RuntimeError("mixed decoder inventory is not a PASS for the pinned GGUF")
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-full64-one-token-oracle-v2" or not oracle.get("captured_complete_model"):
        raise RuntimeError("full64 boundary oracle is incomplete")
    reference = oracle["checkpoints"]
    token_id = int(oracle["token_id"])

    for layer in range(base.DECODER_LAYERS):
        if layer % 4 == 3:
            vref = reference.get(f"Vcur-{layer}")
            if vref is None or len(vref) != attn.KV_DIM:
                raise RuntimeError(f"oracle missing Vcur-{layer}")

    directory = parse_gguf(model)
    trunk = work_dir / "decoder64.k3.bin"
    manifest_path = work_dir / "decoder64.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, manifest_path, layers=range(base.DECODER_LAYERS),
        model_id=gdn.MODEL_ID, revision=gdn.REVISION, source_sha256=gdn.SHA256,
        expected_layers=base.DECODER_LAYERS,
    )
    max_layer_bytes = max(int(x["read_bytes"]) for x in manifest["layers"])
    runtime = gdn.QuantRuntime(gdn._load_native(native_lib))
    free_hidden = gdn._embedding_row(model, directory, token_id)

    free_layer_metrics: dict[str, Any] = {}
    local_layer_metrics: dict[str, Any] = {}
    full_attention_boundaries: dict[str, Any] = {}
    first_free_drift_layer: int | None = None
    first_bad_local_layer: int | None = None
    first_native_cache_cliff_layer: int | None = None
    full_kv_bytes = 0

    with K3Trunk(
        trunk, manifest_path, budget_bytes=2 * max_layer_bytes, want_ring=2,
        max_pinned=0, prefer_direct_io=True,
    ) as reader:
        for layer in range(base.DECODER_LAYERS):
            bound = reader.bind(layer)
            if layer + 1 < base.DECODER_LAYERS:
                reader.prefetch(layer + 1)
            metas = base._layer_meta(manifest, layer)
            prefix = f"blk.{layer}"
            def view(suffix: str): return reader.tensor_view(bound, f"{prefix}.{suffix}")
            def vec(suffix: str): return gdn.f32_vector(view(suffix))

            kind = "full_attention" if layer % 4 == 3 else "gated_deltanet"
            if layer % 4 == 3:
                free_out, kv_bytes, _ = run_full_attn(runtime, view, metas, vec, free_hidden, layer)
                full_kv_bytes += kv_bytes
            else:
                free_out = base._run_recurrent_layer(runtime, view, metas, vec, free_hidden, layer)

            if layer == 0:
                local_out = free_out
                local_native_out = free_out
                local_input_source = "model_input_embedding"
                boundary_info = None
            else:
                local_input = reference.get(f"post_ffn-{layer - 1}")
                if local_input is None:
                    raise RuntimeError(f"oracle missing post_ffn-{layer - 1}")
                local_input_source = f"oracle_post_ffn-{layer - 1}"
                if layer % 4 == 3:
                    local_native_out, _, v_native = run_full_attn(runtime, view, metas, vec, local_input, layer)
                    v_ref = reference[f"Vcur-{layer}"]
                    local_out, _, _ = run_full_attn(
                        runtime, view, metas, vec, local_input, layer, v_override=v_ref
                    )
                    v_m = base.metrics(v_ref, v_native)
                    native_post_m = base.metrics(reference[f"post_ffn-{layer}"], local_native_out)
                    boundary_info = {
                        "Vcur_native": v_m,
                        "Vcur_limit": {"max_abs": VCUR_LIMIT[0], "relative_l2": VCUR_LIMIT[1]},
                        "native_post_ffn": native_post_m,
                        "semantic_input": "oracle_Vcur_F32_then_default_F16_cache",
                    }
                    full_attention_boundaries[str(layer)] = boundary_info
                    if first_native_cache_cliff_layer is None and base._over_limit(native_post_m, LOCAL_LAYER_LIMIT):
                        first_native_cache_cliff_layer = layer
                else:
                    local_out = base._run_recurrent_layer(runtime, view, metas, vec, local_input, layer)
                    local_native_out = local_out
                    boundary_info = None

            ref = reference[f"post_ffn-{layer}"]
            free_m = base.metrics(ref, free_out)
            local_m = base.metrics(ref, local_out)
            free_m["kind"] = kind
            local_m["kind"] = kind
            local_m["input_source"] = local_input_source
            if boundary_info is not None:
                local_m["quantized_boundary"] = "oracle_Vcur_before_F16_cache"
            free_layer_metrics[str(layer)] = free_m
            local_layer_metrics[str(layer)] = local_m

            if first_free_drift_layer is None and base._over_limit(free_m, FREE_DRIFT_LAYER_LIMIT):
                first_free_drift_layer = layer
            local_bad = base._over_limit(local_m, LOCAL_LAYER_LIMIT)
            if layer % 4 == 3 and boundary_info is not None:
                local_bad = local_bad or base._over_limit(boundary_info["Vcur_native"], VCUR_LIMIT)
            if first_bad_local_layer is None and local_bad:
                first_bad_local_layer = layer

            free_hidden = free_out
            bound.release()
        reader_report = reader.report()

    tensors = directory.by_name()
    output_norm_w = base._read_f32_tensor(model, tensors["output_norm.weight"])
    free_result_norm = gdn.rms_norm(free_hidden, output_norm_w)
    free_logits = base._stream_q8_logits(model, tensors["output.weight"], runtime, free_result_norm)
    candidate_top10 = base._topk(free_logits, 10)
    top_token = int(candidate_top10[0]["token"])

    local_result_norm = gdn.rms_norm(reference["post_ffn-63"], output_norm_w)
    local_result_norm_m = base.metrics(reference["result_norm"], local_result_norm)
    local_logits = base._stream_q8_logits(model, tensors["output.weight"], runtime, reference["result_norm"])
    local_result_output_m = base.metrics(reference["result_output"], local_logits)

    oracle_top10 = base._topk(reference["result_output"], 10)
    oracle_top = int(oracle_top10[0]["token"])
    top5_overlap = len(
        {int(x["token"]) for x in candidate_top10[:5]}
        & {int(x["token"]) for x in oracle_top10[:5]}
    )
    free_final_metrics = {
        "post_ffn-63": base.metrics(reference["post_ffn-63"], free_hidden),
        "result_norm": base.metrics(reference["result_norm"], free_result_norm),
        "result_output": base.metrics(reference["result_output"], free_logits),
    }

    local_failures: list[str] = []
    if first_bad_local_layer is not None:
        local_failures.append(f"local semantic/boundary failure at layer {first_bad_local_layer}")
    if base._over_limit(local_result_norm_m, base.LOCAL_FINAL_LIMITS["result_norm"]):
        local_failures.append("local result_norm exceeds proven limit")
    if base._over_limit(local_result_output_m, base.LOCAL_FINAL_LIMITS["result_output"]):
        local_failures.append("local result_output exceeds proven limit")

    behavioral_failures = [] if top_token == oracle_top else [f"free top_token={top_token} oracle={oracle_top}"]
    free_warnings: list[str] = []
    if first_free_drift_layer is not None:
        free_warnings.append(f"free numerical drift first exceeds reference limit at layer {first_free_drift_layer}")
    for name, m in free_final_metrics.items():
        if base._over_limit(m, base.FREE_FINAL_REFERENCE_LIMITS[name]):
            free_warnings.append(f"free {name} exceeds reference numerical limit")

    failures = local_failures + behavioral_failures
    recurrent_layers = sum(1 for i in range(base.DECODER_LAYERS) if i % 4 != 3)
    recurrent_state_bytes_f32 = recurrent_layers * gdn.V_HEADS * gdn.HEAD_DIM * gdn.HEAD_DIM * 4
    conv_history_bytes_f32 = recurrent_layers * (gdn.CONV_KERNEL - 1) * gdn.CONV_DIM * 4

    result = {
        "schema": "qwen38-k3-full64-quantized-boundary-v1",
        "status": "PASS" if not failures else "FAIL",
        "failure_class": None if not failures else "model correctness",
        "model_sha256": gdn.SHA256,
        "llama_cpp_reference_revision": gdn.LLAMA_CPP_REVISION,
        "token_id": token_id,
        "position": 0,
        "validation_model": {
            "free_running": "native activations chained through all 64 layers; no oracle injection",
            "local_recurrent": "oracle previous-layer post_ffn input",
            "local_full_attention": "oracle previous-layer post_ffn plus native Vcur projection check; oracle Vcur injected immediately before F16 cache",
            "reason": "F16 cache quantization is discontinuous; the layer59 microscope proved a 2.17e-7 Vcur projection difference can cross one binary16 bin",
        },
        "local_semantic_status": "PASS" if not local_failures else "FAIL",
        "local_layer_limit": {"max_abs": LOCAL_LAYER_LIMIT[0], "relative_l2": LOCAL_LAYER_LIMIT[1]},
        "Vcur_projection_limit": {"max_abs": VCUR_LIMIT[0], "relative_l2": VCUR_LIMIT[1]},
        "local_layer_metrics": local_layer_metrics,
        "full_attention_boundaries": full_attention_boundaries,
        "first_native_cache_cliff_layer": first_native_cache_cliff_layer,
        "first_bad_local_layer": first_bad_local_layer,
        "free_layer_metrics": free_layer_metrics,
        "first_free_drift_layer": first_free_drift_layer,
        "local_final_metrics": {
            "result_norm": local_result_norm_m,
            "result_output": local_result_output_m,
        },
        "free_final_metrics": free_final_metrics,
        "behavior": {
            "candidate_top_token": top_token,
            "oracle_top_token": oracle_top,
            "top5_overlap": top5_overlap,
            "candidate_top10": candidate_top10,
            "oracle_top10": oracle_top10,
        },
        "state_budget_first_token": {
            "full_attention_kv_bytes_f16": full_kv_bytes,
            "future_recurrent_state_bytes_f32": recurrent_state_bytes_f32,
            "future_conv_history_bytes_f32": conv_history_bytes_f32,
        },
        "candidate": {
            "storage": "K3Trunk two-slot streaming",
            "native_quant_matvec": True,
            "full_matrix_dequantized": False,
            "activation_quantizations": runtime.activation_quantizations,
            "matvec_rows": runtime.matvec_rows,
            "reader_report": reader_report,
        },
        "free_numerical_warnings": free_warnings,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": base.rss_gib(),
    }
    base.atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    return result


def sanity() -> None:
    exact.sanity()
    full_layers = [i for i in range(base.DECODER_LAYERS) if i % 4 == 3]
    if len(full_layers) != 16 or full_layers[0] != 3 or full_layers[-1] != 63:
        raise SystemExit("full-attention schedule sanity failed")
    if VCUR_LIMIT != (5e-4, 2e-4):
        raise SystemExit("Vcur threshold drifted from proven layer3 gate")
    print(json.dumps({
        "schema": "qwen38-k3-full64-quantized-boundary-sanity-v1",
        "status": "PASS", "full_attention_layers": full_layers,
        "Vcur_limit": VCUR_LIMIT,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--oracle", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "sanity": sanity()
    else:
        a.work_dir.mkdir(parents=True, exist_ok=True)
        execute(a.model, a.native_lib, a.inventory, a.oracle, a.work_dir, a.output)


if __name__ == "__main__":
    main()
