#!/usr/bin/env python3
"""Bounded-RAM full-64 one-token executor for Ternary Bonsai 2 27B.

Correctness lanes:
- local/teacher-forced: each layer consumes the pinned Prism previous-layer
  post_ffn checkpoint, isolating that layer from upstream drift;
- free-running: native outputs are chained through all 64 layers.

Weights remain PTQ1_0/PQ2_0 in the K3 two-slot ring. Folded activations use the
Prism explicit-sign + normalized Hadamard transform in native C. The inverse
token-embedding transform, Qwen3.5 GDN ssm_out regroup, GGML-compatible BF16
alpha/beta path, and persistent row-parallel low-bit kernels are all supplied
by bonsai2_quant_runtime.py.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime  # noqa: E402
from gguf_k3_layout import pack_gguf_layers  # noqa: E402
from gguf_stream import parse_gguf  # noqa: E402
from k3_stream import K3Trunk  # noqa: E402
import qwen35_gdn_quant_layer_gate as gdn  # noqa: E402
import qwen35_full_attn_layer3_gate as attn  # noqa: E402

MODEL_ID = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
MODEL_SHA256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
PRISM_REVISION = "d8f26eec76da6d09bb708bcba51ef64b8cd868a3"

N_LAYER = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
LOWBIT = {"PTQ1_0": (128, 28), "PQ2_0": (128, 34)}
LM_HEAD_CHUNK_ROWS = 4096

# Deliberately diagnostic-first. Layer 0 is already much tighter (~1e-7 rel).
# The wider limits here let the first run expose layer-3/full-attention behavior
# and cumulative low-bit quantizer cliffs rather than failing without evidence.
LOCAL_LAYER_LIMIT = (2e-2, 5e-3)
LOCAL_FINAL_NORM_LIMIT = (1e-2, 3e-3)
LOCAL_FINAL_LOGIT_LIMIT = (5e-2, 5e-3)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


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


def over_limit(m: dict[str, float], limit: tuple[float, float]) -> bool:
    return m.get("max_abs", math.inf) > limit[0] or m.get("relative_l2", math.inf) > limit[1]


def topk(values: Sequence[float], k: int = 10) -> list[dict[str, float | int]]:
    ids = heapq.nlargest(min(k, len(values)), range(len(values)), key=values.__getitem__)
    return [{"token": int(i), "logit": float(values[i])} for i in ids]


def layer_meta(manifest: dict[str, Any], layer: int) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == layer)
    return {t["name"]: t for t in entry["tensors"]}


def read_f32_global(model: Path, tensor) -> list[float]:
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


def embedding_row(
    model: Path,
    directory,
    runtime: Bonsai2NativeRuntime,
    token_id: int,
) -> list[float]:
    tensor = directory.by_name()["token_embd.weight"]
    ne0, rows = map(int, tensor.shape)
    if tensor.type_name not in LOWBIT or ne0 != HIDDEN or not 0 <= token_id < rows:
        raise ValueError(
            f"unexpected token embedding type={tensor.type_name} "
            f"shape={list(tensor.shape)} token={token_id}"
        )
    qk, block = LOWBIT[tensor.type_name]
    row_bytes = (ne0 // qk) * block
    if row_bytes * rows != tensor.nbytes:
        raise ValueError("token embedding row geometry mismatch")
    fd = os.open(model, os.O_RDONLY)
    try:
        raw = os.pread(fd, row_bytes, tensor.data_offset + token_id * row_bytes)
    finally:
        os.close(fd)
    if len(raw) != row_bytes:
        raise EOFError("short token embedding row")
    return runtime.dequantize_lookup_row(raw, tensor.type_name, ne0, tensor.name)


def run_ffn(runtime, view, metas, prefix: str, x: Sequence[float]) -> list[float]:
    prepared = runtime.prepare_activation(f"{prefix}.ffn_gate.weight", x)
    gate = runtime.matvec_prepared(
        view("ffn_gate.weight"), metas[f"{prefix}.ffn_gate.weight"], prepared
    )
    up = runtime.matvec_prepared(
        view("ffn_up.weight"), metas[f"{prefix}.ffn_up.weight"], prepared
    )
    swiglu = [gdn.silu(gate[i]) * up[i] for i in range(INTERMEDIATE)]
    return runtime.matvec(
        view("ffn_down.weight"), metas[f"{prefix}.ffn_down.weight"], swiglu
    )


def run_recurrent_layer(runtime, view, metas, vec, hidden: Sequence[float], layer: int) -> list[float]:
    p = f"blk.{layer}"
    x = runtime.rms_norm(hidden, vec("attn_norm.weight"), eps=gdn.RMS_EPS)

    # qkv/z share the same 5120-wide folded activation.
    prepared = runtime.prepare_activation(f"{p}.attn_qkv.weight", x)
    qkv = runtime.matvec_prepared(
        view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], prepared
    )
    z = runtime.matvec_prepared(
        view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], prepared
    )

    # Prism/GGML BF16 semantics are handled inside runtime.matvec.
    beta_raw = runtime.matvec(
        view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], x
    )
    alpha = runtime.matvec(
        view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], x
    )
    beta = [gdn.sigmoid(v) for v in beta_raw]

    # First token starts from zero recurrent state. Evaluate the decay contract
    # even though it cannot affect the zero-state one-token output.
    dt = vec("ssm_dt.bias")
    aa = vec("ssm_a")
    decay = [aa[i] * gdn.softplus(alpha[i] + dt[i]) for i in range(gdn.V_HEADS)]
    if not all(math.isfinite(v) and v <= 0.0 for v in decay):
        raise ValueError(f"layer {layer}: invalid GDN decay")

    kernels = vec("ssm_conv1d.weight")
    if len(kernels) != gdn.CONV_DIM * gdn.CONV_KERNEL:
        raise ValueError(f"layer {layer}: conv kernel shape mismatch")
    conv = [
        gdn.silu(qkv[c] * kernels[c * gdn.CONV_KERNEL + (gdn.CONV_KERNEL - 1)])
        for c in range(gdn.CONV_DIM)
    ]
    q = conv[: gdn.KEY_DIM]
    k = conv[gdn.KEY_DIM : 2 * gdn.KEY_DIM]
    v = conv[2 * gdn.KEY_DIM :]

    core = gdn.one_token_core(q, k, v, beta)
    norm_w = vec("ssm_norm.weight")
    core_heads = gdn.split_heads(core, gdn.V_HEADS)
    z_heads = gdn.split_heads(z, gdn.V_HEADS)
    gated: list[list[float]] = []
    for ch, zh in zip(core_heads, z_heads):
        inv = 1.0 / math.sqrt(
            math.fsum(t * t for t in ch) / gdn.HEAD_DIM + gdn.RMS_EPS
        )
        gated.append([
            ch[d] * inv * norm_w[d] * gdn.silu(zh[d])
            for d in range(gdn.HEAD_DIM)
        ])

    # The runtime recognizes ".ssm_out." and performs Prism's Qwen3.5
    # tiled->grouped 48-head permutation before signs + Hadamard.
    attn_out = runtime.matvec(
        view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gdn.flatten(gated)
    )
    residual = [float(hidden[i]) + attn_out[i] for i in range(HIDDEN)]
    post = runtime.rms_norm(residual, vec("post_attention_norm.weight"), eps=gdn.RMS_EPS)
    ffn = run_ffn(runtime, view, metas, p, post)
    return [residual[i] + ffn[i] for i in range(HIDDEN)]


def run_full_attention_layer(
    runtime, view, metas, vec, hidden: Sequence[float], layer: int
) -> tuple[list[float], int]:
    p = f"blk.{layer}"
    x = runtime.rms_norm(hidden, vec("attn_norm.weight"), eps=gdn.RMS_EPS)

    # q/k/v are all folded from the same 5120-wide activation. The explicit
    # Prism sign vector is width-keyed, so a single transformed+Q8 activation is
    # correct for all three projections.
    prepared = runtime.prepare_activation(f"{p}.attn_q.weight", x)
    qg = runtime.matvec_prepared(
        view("attn_q.weight"), metas[f"{p}.attn_q.weight"], prepared
    )
    k = runtime.matvec_prepared(
        view("attn_k.weight"), metas[f"{p}.attn_k.weight"], prepared
    )
    v = runtime.matvec_prepared(
        view("attn_v.weight"), metas[f"{p}.attn_v.weight"], prepared
    )

    q, gate = attn.split_q_gate(qg)
    q_norm = runtime.rms_norm(q, vec("attn_q_norm.weight"), rows=attn.N_HEAD, eps=attn.RMS_EPS)
    k_norm = runtime.rms_norm(k, vec("attn_k_norm.weight"), rows=attn.N_HEAD_KV, eps=attn.RMS_EPS)

    # Position 0 RoPE is identity. Default Prism/llama.cpp cache storage is F16.
    # With one key, softmax is exactly 1 and the pre-gate attention output is
    # simply the GQA-expanded cached V value.
    k_cache = attn.f16_roundtrip(k_norm)
    v_cache = attn.f16_roundtrip(v)
    pregate = attn.gqa_one_key_attention(v_cache)
    gate_sigmoid = [gdn.sigmoid(t) for t in gate]
    gated = [pregate[i] * gate_sigmoid[i] for i in range(attn.Q_DIM)]

    attn_out = runtime.matvec(
        view("attn_output.weight"), metas[f"{p}.attn_output.weight"], gated
    )
    residual = [float(hidden[i]) + attn_out[i] for i in range(HIDDEN)]
    post = runtime.rms_norm(residual, vec("post_attention_norm.weight"), eps=gdn.RMS_EPS)
    ffn = run_ffn(runtime, view, metas, p, post)
    return [residual[i] + ffn[i] for i in range(HIDDEN)], (len(k_cache) + len(v_cache)) * 2


def stream_lowbit_logits(
    model: Path,
    tensor,
    runtime: Bonsai2NativeRuntime,
    hidden: Sequence[float],
    chunk_rows: int = LM_HEAD_CHUNK_ROWS,
) -> list[float]:
    if tensor.type_name not in LOWBIT:
        raise ValueError(f"{tensor.name}: expected PTQ1_0/PQ2_0, got {tensor.type_name}")
    ne0, rows = map(int, tensor.shape)
    if ne0 != HIDDEN or rows != VOCAB:
        raise ValueError(f"unexpected output head shape {list(tensor.shape)}")
    qk, block = LOWBIT[tensor.type_name]
    row_bytes = (ne0 // qk) * block
    if row_bytes * rows != tensor.nbytes:
        raise ValueError("output head row geometry mismatch")

    prepared = runtime.prepare_activation(tensor.name, hidden)
    logits: list[float] = []
    fd = os.open(model, os.O_RDONLY)
    try:
        for row0 in range(0, rows, chunk_rows):
            nrows = min(chunk_rows, rows - row0)
            nbytes = nrows * row_bytes
            raw = bytearray(os.pread(fd, nbytes, tensor.data_offset + row0 * row_bytes))
            if len(raw) != nbytes:
                raise EOFError(f"short output-head read at row {row0}")
            view = memoryview(raw)
            meta = {
                "name": tensor.name,
                "type_name": tensor.type_name,
                "shape": [ne0, nrows],
            }
            logits.extend(runtime.matvec_prepared(view, meta, prepared))
            view.release()
    finally:
        os.close(fd)
    return logits


def execute(
    model: Path,
    native_lib: Path,
    oracle_json: Path,
    work_dir: Path,
    output: Path,
    threads: int,
) -> dict[str, Any]:
    started = time.monotonic()
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if (
        oracle.get("schema") != "qwen38-llama-full64-one-token-oracle-v1"
        or not oracle.get("captured_complete_model")
    ):
        raise RuntimeError(f"Prism full64 oracle incomplete: {oracle.get('error')}")
    reference = oracle["checkpoints"]
    token_id = int(oracle["token_id"])

    directory = parse_gguf(model)
    if directory.metadata.get("general.architecture") != "qwen35":
        raise ValueError("Bonsai 2 GGUF architecture is not qwen35")
    if int(directory.metadata.get("general.file_type", -1)) not in (142, 143):
        raise ValueError(
            f"unexpected Bonsai 2 file type {directory.metadata.get('general.file_type')}"
        )

    runtime = Bonsai2NativeRuntime(
        native_lib, directory.metadata, threads=threads, max_rows=VOCAB
    )
    tensors = directory.by_name()
    free_hidden = embedding_row(model, directory, runtime, token_id)

    work_dir.mkdir(parents=True, exist_ok=True)
    trunk = work_dir / "decoder64.k3.bin"
    manifest_path = work_dir / "decoder64.k3.json"
    pack_t0 = time.monotonic()
    manifest = pack_gguf_layers(
        directory,
        trunk,
        manifest_path,
        layers=range(N_LAYER),
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        source_sha256=MODEL_SHA256,
        expected_layers=N_LAYER,
    )
    pack_seconds = time.monotonic() - pack_t0
    max_layer_bytes = max(int(x["read_bytes"]) for x in manifest["layers"])
    budget = 2 * max_layer_bytes

    free_metrics: dict[str, Any] = {}
    local_metrics: dict[str, Any] = {}
    first_bad_local: int | None = None
    first_free_drift: int | None = None
    local_layer_seconds: dict[str, float] = {}
    free_layer_seconds: dict[str, float] = {}
    kv_bytes = 0

    with K3Trunk(
        trunk,
        manifest_path,
        budget_bytes=budget,
        want_ring=2,
        max_pinned=0,
        prefer_direct_io=True,
    ) as reader:
        for layer in range(N_LAYER):
            bound = reader.bind(layer)
            if layer + 1 < N_LAYER:
                reader.prefetch(layer + 1)
            metas = layer_meta(manifest, layer)
            p = f"blk.{layer}"

            def view(suffix: str) -> memoryview:
                return reader.tensor_view(bound, f"{p}.{suffix}")

            def vec(suffix: str) -> list[float]:
                return gdn.f32_vector(view(suffix))

            kind = "full_attention" if layer % 4 == 3 else "gated_deltanet"

            t0 = time.monotonic()
            if layer % 4 == 3:
                free_out, this_kv = run_full_attention_layer(
                    runtime, view, metas, vec, free_hidden, layer
                )
                kv_bytes += this_kv
            else:
                free_out = run_recurrent_layer(
                    runtime, view, metas, vec, free_hidden, layer
                )
            free_layer_seconds[str(layer)] = time.monotonic() - t0

            if layer == 0:
                local_out = free_out
                local_source = "model_input_embedding"
                local_layer_seconds[str(layer)] = free_layer_seconds[str(layer)]
            else:
                local_input = reference.get(f"post_ffn-{layer - 1}")
                if local_input is None:
                    raise RuntimeError(f"oracle missing post_ffn-{layer - 1}")
                local_source = f"oracle_post_ffn-{layer - 1}"
                t0 = time.monotonic()
                if layer % 4 == 3:
                    local_out, _ = run_full_attention_layer(
                        runtime, view, metas, vec, local_input, layer
                    )
                else:
                    local_out = run_recurrent_layer(
                        runtime, view, metas, vec, local_input, layer
                    )
                local_layer_seconds[str(layer)] = time.monotonic() - t0

            ref = reference.get(f"post_ffn-{layer}")
            if ref is None:
                fm = lm = {
                    "max_abs": math.inf,
                    "rmse": math.inf,
                    "relative_l2": math.inf,
                    "missing": True,
                }
            else:
                fm = metrics(ref, free_out)
                lm = metrics(ref, local_out)
            fm = dict(fm)
            lm = dict(lm)
            fm["kind"] = kind
            lm["kind"] = kind
            lm["input_source"] = local_source
            free_metrics[str(layer)] = fm
            local_metrics[str(layer)] = lm

            if first_bad_local is None and over_limit(lm, LOCAL_LAYER_LIMIT):
                first_bad_local = layer
            if first_free_drift is None and over_limit(fm, LOCAL_LAYER_LIMIT):
                first_free_drift = layer

            # Emit progress for long hosted and laptop runs.
            print(json.dumps({
                "layer": layer,
                "kind": kind,
                "local_rel_l2": lm.get("relative_l2"),
                "free_rel_l2": fm.get("relative_l2"),
                "free_seconds": free_layer_seconds[str(layer)],
                "local_seconds": local_layer_seconds[str(layer)],
            }), flush=True)

            free_hidden = free_out
            bound.release()

        reader_report = reader.report()

    norm_w = read_f32_global(model, tensors["output_norm.weight"])

    # Isolated final-head semantic lane.
    local_norm = runtime.rms_norm(reference["post_ffn-63"], norm_w, eps=gdn.RMS_EPS)
    local_norm_m = metrics(reference["result_norm"], local_norm)
    local_logits = stream_lowbit_logits(
        model, tensors["output.weight"], runtime, reference["result_norm"]
    )
    local_logits_m = metrics(reference["result_output"], local_logits)

    # Actual free-running result.
    free_norm = runtime.rms_norm(free_hidden, norm_w, eps=gdn.RMS_EPS)
    free_norm_m = metrics(reference["result_norm"], free_norm)
    free_logits = stream_lowbit_logits(
        model, tensors["output.weight"], runtime, free_norm
    )
    free_logits_m = metrics(reference["result_output"], free_logits)

    candidate_top10 = topk(free_logits, 10)
    oracle_top10 = topk(reference["result_output"], 10)
    top_token = int(candidate_top10[0]["token"])
    oracle_top = int(oracle_top10[0]["token"])

    local_failures: list[str] = []
    if first_bad_local is not None:
        m = local_metrics[str(first_bad_local)]
        local_failures.append(
            f"layer {first_bad_local}: max_abs={m['max_abs']:.6g}, "
            f"relative_l2={m['relative_l2']:.6g}"
        )
    if over_limit(local_norm_m, LOCAL_FINAL_NORM_LIMIT):
        local_failures.append(
            f"result_norm: max_abs={local_norm_m['max_abs']:.6g}, "
            f"relative_l2={local_norm_m['relative_l2']:.6g}"
        )
    if over_limit(local_logits_m, LOCAL_FINAL_LOGIT_LIMIT):
        local_failures.append(
            f"result_output: max_abs={local_logits_m['max_abs']:.6g}, "
            f"relative_l2={local_logits_m['relative_l2']:.6g}"
        )

    behavioral_failures = []
    if top_token != oracle_top:
        behavioral_failures.append(f"top_token native={top_token} prism={oracle_top}")

    result = {
        "schema": "qwen38-bonsai2-full64-one-token-v1",
        "status": "PASS" if not (local_failures or behavioral_failures) else "FAIL",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "prism_revision": PRISM_REVISION,
        "token_id": token_id,
        "threads": threads,
        "decoder_layers": N_LAYER,
        "recurrent_layers": 48,
        "full_attention_layers": 16,
        "local_layer_limit": {
            "max_abs": LOCAL_LAYER_LIMIT[0],
            "relative_l2": LOCAL_LAYER_LIMIT[1],
        },
        "local_layer_metrics": local_metrics,
        "free_layer_metrics": free_metrics,
        "first_bad_local_layer": first_bad_local,
        "first_free_drift_layer": first_free_drift,
        "local_final": {
            "result_norm": local_norm_m,
            "result_output": local_logits_m,
        },
        "free_final": {
            "post_ffn_63": metrics(reference["post_ffn-63"], free_hidden),
            "result_norm": free_norm_m,
            "result_output": free_logits_m,
        },
        "top_token": top_token,
        "oracle_top_token": oracle_top,
        "top10": candidate_top10,
        "oracle_top10": oracle_top10,
        "top5_overlap": len(
            {int(x["token"]) for x in candidate_top10[:5]}
            & {int(x["token"]) for x in oracle_top10[:5]}
        ),
        "local_failures": local_failures,
        "behavioral_failures": behavioral_failures,
        "timing": {
            "pack_seconds": pack_seconds,
            "free_layer_seconds": free_layer_seconds,
            "local_layer_seconds": local_layer_seconds,
            "free_layers_total_seconds": sum(free_layer_seconds.values()),
            "local_layers_total_seconds": sum(local_layer_seconds.values()),
            "elapsed_seconds": time.monotonic() - started,
        },
        "kv_cache_bytes_f16_one_token": kv_bytes,
        "reader": reader_report,
        "runtime": runtime.report(),
        "max_rss_gib": rss_gib(),
    }
    runtime.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "first_bad_local_layer": first_bad_local,
        "first_free_drift_layer": first_free_drift,
        "top_token": top_token,
        "oracle_top_token": oracle_top,
        "top5_overlap": result["top5_overlap"],
        "local_final": result["local_final"],
        "free_final": result["free_final"],
        "timing": result["timing"],
        "reader": reader_report,
        "runtime": result["runtime"],
        "max_rss_gib": result["max_rss_gib"],
        "local_failures": local_failures,
        "behavioral_failures": behavioral_failures,
    }, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)
    print("QWEN38_BONSAI2_FULL64_ONE_TOKEN_PASS")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--native-lib", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    execute(
        args.model,
        args.native_lib,
        args.oracle,
        args.work_dir,
        args.output,
        args.threads,
    )


if __name__ == "__main__":
    main()
