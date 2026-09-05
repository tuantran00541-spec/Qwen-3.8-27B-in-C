#!/usr/bin/env python3
"""Bounded-RAM one-token full-model executor for Qwen3.8-27B Q6_K_L.

This composes the already-proven recurrent layer-0 and full-attention layer-3
semantics across decoder blocks 0..63. It is deliberately the first-token
milestone: every recurrent layer starts from zero state/history, and each full
attention layer sees one key at position 0. The executor records the F16 K/V
state representation needed by future multi-token decoding.

The real gate emits two independent evidence lanes:

* local semantics: each layer > 0 is evaluated with the pinned llama.cpp
  post_ffn output from the previous layer. This removes upstream activation-
  quantizer cliffs and asks whether that layer itself implements the pinned
  graph correctly.
* free running: the custom K3 output is chained through all 64 layers exactly
  as a real decoder would run. Numerical drift remains fully reported and the
  emitted top token must match llama.cpp.

Weights are packed once into the existing K3 trunk and streamed with two ring
slots. The global Q8_0 LM head is never fully dequantized; logits are evaluated
in bounded row chunks directly from the pinned GGUF.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
from gguf_k3_layout import pack_gguf_layers
from gguf_quant_ref import row_nbytes
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

DECODER_LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
LM_HEAD_CHUNK_ROWS = 4096
LOCAL_LAYER_LIMIT = (6e-3, 2e-3)
FREE_DRIFT_LAYER_LIMIT = (6e-3, 2e-3)
LOCAL_FINAL_LIMITS = {
    "result_norm": (3e-3, 1e-3),
    "result_output": (2e-2, 2e-3),
}
FREE_FINAL_REFERENCE_LIMITS = {
    "post_ffn-63": (6e-3, 2e-3),
    "result_norm": (3e-3, 1e-3),
    "result_output": (2e-2, 2e-3),
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def metrics(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    if len(ref) != len(cand):
        return {
            "length_mismatch": float(abs(len(ref) - len(cand))),
            "max_abs": math.inf,
            "rmse": math.inf,
            "relative_l2": math.inf,
        }
    diffs = [float(b) - float(a) for a, b in zip(ref, cand)]
    err2 = math.fsum(v * v for v in diffs)
    ref2 = math.fsum(float(v) * float(v) for v in ref)
    return {
        "max_abs": max((abs(v) for v in diffs), default=0.0),
        "rmse": math.sqrt(err2 / max(1, len(diffs))),
        "relative_l2": math.sqrt(err2 / ref2) if ref2 else math.sqrt(err2),
    }


def _topk(values: Sequence[float], k: int = 10) -> list[dict[str, float | int]]:
    if k <= 0:
        return []
    ids = heapq.nlargest(min(k, len(values)), range(len(values)), key=values.__getitem__)
    return [{"token": int(i), "logit": float(values[i])} for i in ids]


def _over_limit(m: dict[str, float], limit: tuple[float, float]) -> bool:
    return m.get("max_abs", math.inf) > limit[0] or m.get("relative_l2", math.inf) > limit[1]


def _layer_meta(manifest: dict[str, Any], layer: int) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == layer)
    return {t["name"]: t for t in entry["tensors"]}


def _read_f32_tensor(model: Path, tensor) -> list[float]:
    if tensor.type_name != "F32":
        raise ValueError(f"{tensor.name}: expected F32, got {tensor.type_name}")
    fd = os.open(model, os.O_RDONLY)
    try:
        raw = os.pread(fd, tensor.nbytes, tensor.data_offset)
    finally:
        os.close(fd)
    if len(raw) != tensor.nbytes:
        raise EOFError(f"short read for {tensor.name}")
    return gdn.f32_vector(memoryview(raw))


def _run_ffn(runtime: gdn.QuantRuntime, view, metas: dict[str, dict[str, Any]],
             prefix: str, post_norm: Sequence[float]) -> list[float]:
    prepared = runtime.quantize(post_norm, "Q6_K")
    gate = runtime.matvec(view("ffn_gate.weight"), metas[f"{prefix}.ffn_gate.weight"], post_norm, prepared)
    up = runtime.matvec(view("ffn_up.weight"), metas[f"{prefix}.ffn_up.weight"], post_norm, prepared)
    swiglu = [gdn.silu(gate[i]) * up[i] for i in range(INTERMEDIATE)]
    return runtime.matvec(view("ffn_down.weight"), metas[f"{prefix}.ffn_down.weight"], swiglu)


def _run_recurrent_layer(runtime: gdn.QuantRuntime, view, metas: dict[str, dict[str, Any]],
                         vec, hidden: Sequence[float], layer: int) -> list[float]:
    prefix = f"blk.{layer}"
    attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qkv = runtime.matvec(view("attn_qkv.weight"), metas[f"{prefix}.attn_qkv.weight"], attn_norm)
    z = runtime.matvec(view("attn_gate.weight"), metas[f"{prefix}.attn_gate.weight"], attn_norm)
    beta_raw = runtime.matvec(view("ssm_beta.weight"), metas[f"{prefix}.ssm_beta.weight"], attn_norm)
    alpha = runtime.matvec(view("ssm_alpha.weight"), metas[f"{prefix}.ssm_alpha.weight"], attn_norm)
    beta = [gdn.sigmoid(x) for x in beta_raw]

    # Zero initial recurrent state makes the first-token decay irrelevant to the
    # output, but evaluating it locks the future state-transition contract.
    dt = vec("ssm_dt.bias")
    a = vec("ssm_a")
    decay = [a[i] * gdn.softplus(alpha[i] + dt[i]) for i in range(gdn.V_HEADS)]
    if not all(math.isfinite(x) and x <= 0.0 for x in decay):
        raise ValueError(f"layer {layer}: invalid recurrent decay")

    kernels = vec("ssm_conv1d.weight")
    if len(kernels) != gdn.CONV_DIM * gdn.CONV_KERNEL:
        raise ValueError(f"layer {layer}: conv kernel shape mismatch")
    conv = [
        gdn.silu(qkv[c] * kernels[c * gdn.CONV_KERNEL + (gdn.CONV_KERNEL - 1)])
        for c in range(gdn.CONV_DIM)
    ]
    q = conv[:gdn.KEY_DIM]
    k = conv[gdn.KEY_DIM:2 * gdn.KEY_DIM]
    v = conv[2 * gdn.KEY_DIM:]

    core = gdn.one_token_core(q, k, v, beta)
    norm_w = vec("ssm_norm.weight")
    core_heads = gdn.split_heads(core, gdn.V_HEADS)
    z_heads = gdn.split_heads(z, gdn.V_HEADS)
    gated: list[list[float]] = []
    for ch, zh in zip(core_heads, z_heads):
        inv_rms = 1.0 / math.sqrt(math.fsum(x * x for x in ch) / gdn.HEAD_DIM + gdn.RMS_EPS)
        gated.append([
            ch[d] * inv_rms * norm_w[d] * gdn.silu(zh[d])
            for d in range(gdn.HEAD_DIM)
        ])
    linear_out = runtime.matvec(
        view("ssm_out.weight"), metas[f"{prefix}.ssm_out.weight"], gdn.flatten(gated)
    )
    residual = [float(hidden[i]) + linear_out[i] for i in range(HIDDEN)]
    post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    ffn_out = _run_ffn(runtime, view, metas, prefix, post_norm)
    return [residual[i] + ffn_out[i] for i in range(HIDDEN)]


def _run_full_attn_layer(runtime: gdn.QuantRuntime, view, metas: dict[str, dict[str, Any]],
                         vec, hidden: Sequence[float], layer: int) -> tuple[list[float], int]:
    prefix = f"blk.{layer}"
    attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qg = runtime.matvec(view("attn_q.weight"), metas[f"{prefix}.attn_q.weight"], attn_norm)
    q, gate = attn.split_q_gate(qg)
    _q_norm = attn.rms_norm_heads(q, attn.N_HEAD, vec("attn_q_norm.weight"))
    k = runtime.matvec(view("attn_k.weight"), metas[f"{prefix}.attn_k.weight"], attn_norm)
    v = runtime.matvec(view("attn_v.weight"), metas[f"{prefix}.attn_v.weight"], attn_norm)
    k_norm = attn.rms_norm_heads(k, attn.N_HEAD_KV, vec("attn_k_norm.weight"))

    # Position 0 RoPE is identity. K/V are stored to the default F16 cache and
    # read back before attention. With one key, softmax is mathematically 1.
    k_cache = attn.f16_roundtrip(k_norm)
    v_cache = attn.f16_roundtrip(v)
    pregate = attn.gqa_one_key_attention(v_cache)
    gate_sigmoid = [gdn.sigmoid(x) for x in gate]
    gated = [pregate[i] * gate_sigmoid[i] for i in range(attn.Q_DIM)]
    attn_out = runtime.matvec(
        view("attn_output.weight"), metas[f"{prefix}.attn_output.weight"], gated
    )
    residual = [float(hidden[i]) + attn_out[i] for i in range(HIDDEN)]
    post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    ffn_out = _run_ffn(runtime, view, metas, prefix, post_norm)
    final = [residual[i] + ffn_out[i] for i in range(HIDDEN)]
    return final, (len(k_cache) + len(v_cache)) * 2


def _stream_q8_logits(model: Path, tensor, runtime: gdn.QuantRuntime,
                      hidden: Sequence[float], chunk_rows: int = LM_HEAD_CHUNK_ROWS) -> list[float]:
    if tensor.type_name != "Q8_0":
        raise ValueError(f"{tensor.name}: expected Q8_0, got {tensor.type_name}")
    ne0, rows = map(int, tensor.shape)
    if ne0 != HIDDEN or rows != VOCAB:
        raise ValueError(f"unexpected LM head shape {list(tensor.shape)}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    stride = row_nbytes("Q8_0", ne0)
    if stride * rows != tensor.nbytes:
        raise ValueError("LM head Q8_0 byte geometry mismatch")

    prepared = runtime.quantize(hidden, "Q8_0")
    logits: list[float] = []
    fd = os.open(model, os.O_RDONLY)
    try:
        for row0 in range(0, rows, chunk_rows):
            nrows = min(chunk_rows, rows - row0)
            nbytes = nrows * stride
            raw = bytearray(os.pread(fd, nbytes, tensor.data_offset + row0 * stride))
            if len(raw) != nbytes:
                raise EOFError(f"short LM-head read at row {row0}")
            weight_view = memoryview(raw)
            meta = {"name": tensor.name, "type_name": "Q8_0", "shape": [ne0, nrows]}
            logits.extend(runtime.matvec(weight_view, meta, hidden, prepared))
            weight_view.release()
    finally:
        os.close(fd)
    return logits


def execute(model: Path, native_lib: Path, inventory_json: Path, oracle_json: Path,
            work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    inv = json.loads(inventory_json.read_text(encoding="utf-8"))
    if inv.get("status") != "PASS" or inv.get("sha256") != gdn.SHA256:
        raise RuntimeError("mixed decoder inventory is not a PASS for the pinned GGUF")
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-full64-one-token-oracle-v1" or not oracle.get("captured_complete_model"):
        raise RuntimeError("full64 llama oracle is incomplete")
    reference = oracle["checkpoints"]
    token_id = int(oracle["token_id"])

    directory = parse_gguf(model)
    trunk = work_dir / "decoder64.k3.bin"
    manifest_path = work_dir / "decoder64.k3.json"
    manifest = pack_gguf_layers(
        directory,
        trunk,
        manifest_path,
        layers=range(DECODER_LAYERS),
        model_id=gdn.MODEL_ID,
        revision=gdn.REVISION,
        source_sha256=gdn.SHA256,
        expected_layers=DECODER_LAYERS,
    )
    max_layer_bytes = max(int(x["read_bytes"]) for x in manifest["layers"])
    budget = 2 * max_layer_bytes
    runtime = gdn.QuantRuntime(gdn._load_native(native_lib))
    free_hidden = gdn._embedding_row(model, directory, token_id)

    free_layer_metrics: dict[str, Any] = {}
    local_layer_metrics: dict[str, Any] = {}
    first_free_drift_layer: int | None = None
    first_bad_local_layer: int | None = None
    full_kv_bytes = 0

    with K3Trunk(
        trunk,
        manifest_path,
        budget_bytes=budget,
        want_ring=2,
        max_pinned=0,
        prefer_direct_io=True,
    ) as reader:
        for layer in range(DECODER_LAYERS):
            bound = reader.bind(layer)
            if layer + 1 < DECODER_LAYERS:
                reader.prefetch(layer + 1)
            metas = _layer_meta(manifest, layer)
            prefix = f"blk.{layer}"

            def view(suffix: str) -> memoryview:
                return reader.tensor_view(bound, f"{prefix}.{suffix}")

            def vec(suffix: str) -> list[float]:
                return gdn.f32_vector(view(suffix))

            kind = "full_attention" if layer % 4 == 3 else "gated_deltanet"
            if layer % 4 == 3:
                free_out, kv_bytes = _run_full_attn_layer(runtime, view, metas, vec, free_hidden, layer)
                full_kv_bytes += kv_bytes
            else:
                free_out = _run_recurrent_layer(runtime, view, metas, vec, free_hidden, layer)

            # The local lane is teacher-forced only at layer boundaries. Layer 0
            # already starts from the exact model embedding and is reused.
            if layer == 0:
                local_out = free_out
                local_input_source = "model_input_embedding"
            else:
                local_input = reference.get(f"post_ffn-{layer - 1}")
                if local_input is None:
                    raise RuntimeError(f"oracle missing post_ffn-{layer - 1}")
                local_input_source = f"oracle_post_ffn-{layer - 1}"
                if layer % 4 == 3:
                    local_out, _ = _run_full_attn_layer(runtime, view, metas, vec, local_input, layer)
                else:
                    local_out = _run_recurrent_layer(runtime, view, metas, vec, local_input, layer)

            ref = reference.get(f"post_ffn-{layer}")
            if ref is None:
                free_m = {"max_abs": math.inf, "rmse": math.inf, "relative_l2": math.inf, "missing": True}
                local_m = dict(free_m)
            else:
                free_m = metrics(ref, free_out)
                local_m = metrics(ref, local_out)
            free_m["kind"] = kind
            local_m["kind"] = kind
            local_m["input_source"] = local_input_source
            free_layer_metrics[str(layer)] = free_m
            local_layer_metrics[str(layer)] = local_m

            if first_free_drift_layer is None and _over_limit(free_m, FREE_DRIFT_LAYER_LIMIT):
                first_free_drift_layer = layer
            if first_bad_local_layer is None and _over_limit(local_m, LOCAL_LAYER_LIMIT):
                first_bad_local_layer = layer

            free_hidden = free_out
            bound.release()
        reader_report = reader.report()

    tensors = directory.by_name()
    output_norm_w = _read_f32_tensor(model, tensors["output_norm.weight"])

    # Free-running output head: this is the actual token the custom runtime emits.
    free_result_norm = gdn.rms_norm(free_hidden, output_norm_w)
    free_logits = _stream_q8_logits(model, tensors["output.weight"], runtime, free_result_norm)
    candidate_top10 = _topk(free_logits, 10)
    top_token = int(candidate_top10[0]["token"])
    top_logit = float(candidate_top10[0]["logit"])

    # Local output-head semantics are isolated from upstream quantizer cliffs:
    # validate output_norm from exact oracle final hidden, then validate the
    # Q8_0 LM head with the exact oracle result_norm activation.
    local_result_norm = gdn.rms_norm(reference["post_ffn-63"], output_norm_w)
    local_result_norm_m = metrics(reference["result_norm"], local_result_norm)
    local_logits = _stream_q8_logits(model, tensors["output.weight"], runtime, reference["result_norm"])
    local_result_output_m = metrics(reference["result_output"], local_logits)

    free_final_metrics = {
        "post_ffn-63": metrics(reference["post_ffn-63"], free_hidden),
        "result_norm": metrics(reference["result_norm"], free_result_norm),
        "result_output": metrics(reference["result_output"], free_logits),
    }
    oracle_top10 = _topk(reference["result_output"], 10)
    oracle_top = int(oracle_top10[0]["token"])
    candidate_top5 = {int(x["token"]) for x in candidate_top10[:5]}
    oracle_top5 = {int(x["token"]) for x in oracle_top10[:5]}
    top5_overlap = len(candidate_top5 & oracle_top5)
    candidate_margin = float(candidate_top10[0]["logit"] - candidate_top10[1]["logit"])
    oracle_margin = float(oracle_top10[0]["logit"] - oracle_top10[1]["logit"])

    local_failures: list[str] = []
    if first_bad_local_layer is not None:
        m = local_layer_metrics[str(first_bad_local_layer)]
        local_failures.append(
            f"local layer {first_bad_local_layer}: max_abs={m['max_abs']:.6g}>{LOCAL_LAYER_LIMIT[0]:g} "
            f"or rel_l2={m['relative_l2']:.6g}>{LOCAL_LAYER_LIMIT[1]:g}"
        )
    if _over_limit(local_result_norm_m, LOCAL_FINAL_LIMITS["result_norm"]):
        local_failures.append(
            f"local result_norm: max_abs={local_result_norm_m['max_abs']:.6g}>{LOCAL_FINAL_LIMITS['result_norm'][0]:g} "
            f"or rel_l2={local_result_norm_m['relative_l2']:.6g}>{LOCAL_FINAL_LIMITS['result_norm'][1]:g}"
        )
    if _over_limit(local_result_output_m, LOCAL_FINAL_LIMITS["result_output"]):
        local_failures.append(
            f"local result_output: max_abs={local_result_output_m['max_abs']:.6g}>{LOCAL_FINAL_LIMITS['result_output'][0]:g} "
            f"or rel_l2={local_result_output_m['relative_l2']:.6g}>{LOCAL_FINAL_LIMITS['result_output'][1]:g}"
        )

    behavioral_failures: list[str] = []
    if top_token != oracle_top:
        behavioral_failures.append(f"free top_token={top_token} oracle={oracle_top}")

    free_numerical_warnings: list[str] = []
    if first_free_drift_layer is not None:
        m = free_layer_metrics[str(first_free_drift_layer)]
        free_numerical_warnings.append(
            f"free layer {first_free_drift_layer}: max_abs={m['max_abs']:.6g}>{FREE_DRIFT_LAYER_LIMIT[0]:g} "
            f"or rel_l2={m['relative_l2']:.6g}>{FREE_DRIFT_LAYER_LIMIT[1]:g}"
        )
    for name, m in free_final_metrics.items():
        limit = FREE_FINAL_REFERENCE_LIMITS[name]
        if _over_limit(m, limit):
            free_numerical_warnings.append(
                f"free {name}: max_abs={m['max_abs']:.6g}>{limit[0]:g} or rel_l2={m['relative_l2']:.6g}>{limit[1]:g}"
            )

    failures = local_failures + behavioral_failures
    recurrent_layers = sum(1 for i in range(DECODER_LAYERS) if i % 4 != 3)
    recurrent_state_bytes_f32 = recurrent_layers * gdn.V_HEADS * gdn.HEAD_DIM * gdn.HEAD_DIM * 4
    conv_history_bytes_f32 = recurrent_layers * (gdn.CONV_KERNEL - 1) * gdn.CONV_DIM * 4

    result = {
        "schema": "qwen38-k3-full64-one-token-v2",
        "status": "PASS" if not failures else "FAIL",
        "failure_class": None if not failures else "model correctness",
        "model_sha256": gdn.SHA256,
        "official_revision": gdn.REVISION,
        "llama_cpp_reference_revision": gdn.LLAMA_CPP_REVISION,
        "token_id": token_id,
        "position": 0,
        "decoder_layers": DECODER_LAYERS,
        "recurrent_layers": recurrent_layers,
        "full_attention_layers": DECODER_LAYERS - recurrent_layers,
        "validation_model": {
            "local_semantics": "teacher_forced_previous_oracle_post_ffn; layer0 reuses exact embedding path",
            "free_running": "candidate outputs chained through all 64 layers",
            "rationale": "Q8_0 activation quantization is discontinuous; tiny upstream arithmetic differences can change downstream quantized activation codes",
        },
        "local_semantic_status": "PASS" if not local_failures else "FAIL",
        "local_layer_limit": {"max_abs": LOCAL_LAYER_LIMIT[0], "relative_l2": LOCAL_LAYER_LIMIT[1]},
        "local_layer_metrics": local_layer_metrics,
        "first_bad_local_layer": first_bad_local_layer,
        "local_final_metrics": {
            "result_norm_from_oracle_post_ffn63": local_result_norm_m,
            "result_output_from_oracle_result_norm": local_result_output_m,
        },
        "local_final_limits": {k: {"max_abs": v[0], "relative_l2": v[1]} for k, v in LOCAL_FINAL_LIMITS.items()},
        "local_failures": local_failures,
        "free_behavior_status": "PASS" if not behavioral_failures else "FAIL",
        "free_layer_reference_limit": {"max_abs": FREE_DRIFT_LAYER_LIMIT[0], "relative_l2": FREE_DRIFT_LAYER_LIMIT[1]},
        "free_layer_metrics": free_layer_metrics,
        "first_free_drift_layer": first_free_drift_layer,
        "free_final_metrics": free_final_metrics,
        "free_final_reference_limits": {k: {"max_abs": v[0], "relative_l2": v[1]} for k, v in FREE_FINAL_REFERENCE_LIMITS.items()},
        "free_numerical_status": "DRIFT" if free_numerical_warnings else "WITHIN_REFERENCE_LIMITS",
        "free_numerical_warnings": free_numerical_warnings,
        "top_token": top_token,
        "top_logit": top_logit,
        "oracle_top_token": oracle_top,
        "oracle_top_logit": float(oracle_top10[0]["logit"]),
        "candidate_top10": candidate_top10,
        "oracle_top10": oracle_top10,
        "top5_overlap": top5_overlap,
        "candidate_top1_margin": candidate_margin,
        "oracle_top1_margin": oracle_margin,
        "candidate": {
            "storage": "existing K3Trunk 64-layer two-slot stream",
            "native_quant_matvec": True,
            "full_matrix_dequantized": False,
            "lm_head_chunk_rows": LM_HEAD_CHUNK_ROWS,
            "local_teacher_forced_lane": True,
            "free_running_lane": True,
            "activation_quantizations": runtime.activation_quantizations,
            "matvec_rows": runtime.matvec_rows,
            "reader_report": reader_report,
            "packed_file_bytes": int(manifest["packed_file_bytes"]),
            "max_layer_bytes": max_layer_bytes,
        },
        "first_token_state_contract": {
            "full_attention_f16_kv_bytes": full_kv_bytes,
            "future_recurrent_state_bytes_f32": recurrent_state_bytes_f32,
            "future_conv_history_bytes_f32": conv_history_bytes_f32,
        },
        "behavioral_failures": behavioral_failures,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": rss_gib(),
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    return result


def sanity() -> None:
    kinds = ["full" if i % 4 == 3 else "recurrent" for i in range(DECODER_LAYERS)]
    if kinds.count("full") != 16 or kinds.count("recurrent") != 48:
        raise SystemExit("decoder layer-kind schedule sanity failed")
    if [i for i in range(8) if kinds[i] == "full"] != [3, 7]:
        raise SystemExit("full-attention interval sanity failed")

    probe = attn.f16_roundtrip([1.0 / 3.0, -1.0 / 3.0])
    if probe != [0.333251953125, -0.333251953125]:
        raise SystemExit(f"F16 state sanity failed: {probe}")

    top = _topk([0.5, 2.0, 1.0, -3.0], 3)
    if [x["token"] for x in top] != [1, 2, 0]:
        raise SystemExit(f"top-k sanity failed: {top}")

    recurrent_state_bytes = 48 * gdn.V_HEADS * gdn.HEAD_DIM * gdn.HEAD_DIM * 4
    conv_history_bytes = 48 * (gdn.CONV_KERNEL - 1) * gdn.CONV_DIM * 4
    if recurrent_state_bytes <= 0 or conv_history_bytes <= 0:
        raise SystemExit("future recurrent state sizing sanity failed")

    print(json.dumps({
        "schema": "qwen38-k3-full64-one-token-sanity-v2",
        "status": "PASS",
        "decoder_layers": DECODER_LAYERS,
        "recurrent_layers": 48,
        "full_attention_layers": 16,
        "full_attention_ids_head": [3, 7, 11, 15],
        "two_slot_streaming_target": True,
        "lm_head_streamed": True,
        "kv_cache_type_k": "F16",
        "kv_cache_type_v": "F16",
        "local_teacher_forced_lane": True,
        "free_running_lane": True,
        "future_recurrent_state_bytes_f32": recurrent_state_bytes,
        "future_conv_history_bytes_f32": conv_history_bytes,
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--oracle", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if sys.byteorder != "little":
        raise SystemExit("GGUF executor currently requires little-endian host")
    if args.cmd == "sanity":
        sanity()
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        execute(args.model, args.native_lib, args.inventory, args.oracle, args.work_dir, args.output)


if __name__ == "__main__":
    main()
