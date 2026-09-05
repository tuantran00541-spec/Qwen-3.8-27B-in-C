#!/usr/bin/env python3
"""Isolated real layer-3 full-attention semantic gate for Qwen3.8-27B K3.

The pinned llama.cpp oracle supplies only the exact input entering decoder layer 3.
Everything inside layer 3 is executed by the custom bounded-RAM K3 path using the
same native quantized matvec bridge as the proven recurrent layer-0 gate.

For the single BOS token at position 0, RoPE is the identity and attention has
exactly one key.  The softmax is therefore exactly 1 for every query head; the
attention pre-gate output is the GQA-expanded V vector after the default llama.cpp
F16 KV-cache storage round-trip.  Q/K projections and normalization are still
checked against llama.cpp so the full-attention tensor layout is validated rather
than skipped.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as base
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

LAYER = 3
HIDDEN = 5120
INTERMEDIATE = 17408
N_HEAD = 24
N_HEAD_KV = 4
HEAD_DIM = 256
Q_DIM = N_HEAD * HEAD_DIM
KV_DIM = N_HEAD_KV * HEAD_DIM
QG_DIM = 2 * Q_DIM
GQA_REPEAT = N_HEAD // N_HEAD_KV
RMS_EPS = 1e-6


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


def split_heads(values: Sequence[float], heads: int) -> list[list[float]]:
    if len(values) != heads * HEAD_DIM:
        raise ValueError(f"head reshape mismatch: len={len(values)} heads={heads}")
    return [list(map(float, values[h * HEAD_DIM:(h + 1) * HEAD_DIM])) for h in range(heads)]


def flatten(heads: Sequence[Sequence[float]]) -> list[float]:
    return [float(v) for head in heads for v in head]


def split_q_gate(qg: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(qg) != QG_DIM:
        raise ValueError(f"Q/G projection width={len(qg)} expected={QG_DIM}")
    q: list[float] = []
    gate: list[float] = []
    stride = 2 * HEAD_DIM
    for h in range(N_HEAD):
        start = h * stride
        q.extend(map(float, qg[start:start + HEAD_DIM]))
        gate.extend(map(float, qg[start + HEAD_DIM:start + stride]))
    return q, gate


def rms_norm_heads(values: Sequence[float], heads: int, weight: Sequence[float]) -> list[float]:
    if len(weight) != HEAD_DIM:
        raise ValueError("per-head RMSNorm weight width mismatch")
    out: list[list[float]] = []
    for head in split_heads(values, heads):
        inv = 1.0 / math.sqrt(math.fsum(v * v for v in head) / HEAD_DIM + RMS_EPS)
        out.append([head[i] * inv * float(weight[i]) for i in range(HEAD_DIM)])
    return flatten(out)


def f16_roundtrip(values: Sequence[float]) -> list[float]:
    """Round F32 activations through IEEE binary16 like default llama.cpp KV cache."""
    return [struct.unpack("<e", struct.pack("<e", float(v)))[0] for v in values]


def gqa_one_key_attention(v: Sequence[float]) -> list[float]:
    """Expand 4 cached KV heads to 24 query heads for a one-key attention row."""
    kv = split_heads(v, N_HEAD_KV)
    return flatten([kv[qh // GQA_REPEAT] for qh in range(N_HEAD)])


def _layer_meta(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == LAYER)
    return {t["name"]: t for t in entry["tensors"]}


def execute(model: Path, native_lib: Path, inventory_json: Path, oracle_json: Path,
            work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    inv = json.loads(inventory_json.read_text(encoding="utf-8"))
    if inv.get("status") != "PASS" or inv.get("sha256") != base.SHA256:
        raise RuntimeError("mixed-quant decoder inventory is not a PASS for the pinned GGUF")

    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-layer3-oracle-v1" or not oracle.get("captured_complete_layer"):
        raise RuntimeError("llama layer3 oracle is incomplete")
    reference = dict(oracle["checkpoints"])
    hidden = list(map(float, reference["layer3_input"]))
    if len(hidden) != HIDDEN:
        raise ValueError("layer3 oracle input width mismatch")

    # Qcur_reshaped is a non-contiguous view into interleaved Q/G storage.  The
    # optimized oracle callback can observe the aliased physical buffer rather
    # than logical view order, so canonicalize it from the stable Qcur_full
    # checkpoint.  Kcur is similarly optimizer-aliased with Kcur_normed in the
    # pinned graph and is intentionally diagnostic-only; Kcur_normed remains a
    # required semantic checkpoint.
    oracle_q, oracle_gate = split_q_gate(reference["Qcur_full-3"])
    reference["Qcur_reshaped-3"] = oracle_q
    reference["gate_reshaped-3"] = oracle_gate

    directory = parse_gguf(model)
    trunk = work_dir / "layer3.k3.bin"
    manifest_path = work_dir / "layer3.k3.json"
    manifest = pack_gguf_layers(
        directory,
        trunk,
        manifest_path,
        layers=[LAYER],
        model_id=base.MODEL_ID,
        revision=base.REVISION,
        source_sha256=base.SHA256,
        expected_layers=base.DECODER_LAYERS,
    )
    layer_entry = manifest["layers"][0]
    budget = int(layer_entry["read_bytes"])
    runtime = base.QuantRuntime(base._load_native(native_lib))
    checkpoints: dict[str, list[float]] = {"layer3_input": hidden}

    with K3Trunk(trunk, manifest_path, budget_bytes=budget, want_ring=1,
                 max_pinned=0, prefer_direct_io=True) as reader:
        bound = reader.bind(LAYER)
        metas = _layer_meta(manifest)

        def view(suffix: str) -> memoryview:
            return reader.tensor_view(bound, f"blk.{LAYER}.{suffix}")

        def vec(suffix: str) -> list[float]:
            return base.f32_vector(view(suffix))

        attn_norm = base.rms_norm(hidden, vec("attn_norm.weight"))
        checkpoints["attn_norm-3"] = attn_norm

        qg = runtime.matvec(view("attn_q.weight"), metas["blk.3.attn_q.weight"], attn_norm)
        q, gate = split_q_gate(qg)
        checkpoints["Qcur_full-3"] = qg
        checkpoints["Qcur_reshaped-3"] = q
        checkpoints["gate_reshaped-3"] = gate

        q_normed = rms_norm_heads(q, N_HEAD, vec("attn_q_norm.weight"))
        checkpoints["Qcur_normed-3"] = q_normed

        k = runtime.matvec(view("attn_k.weight"), metas["blk.3.attn_k.weight"], attn_norm)
        v = runtime.matvec(view("attn_v.weight"), metas["blk.3.attn_v.weight"], attn_norm)
        checkpoints["Vcur-3"] = v
        k_normed = rms_norm_heads(k, N_HEAD_KV, vec("attn_k_norm.weight"))
        checkpoints["Kcur_normed-3"] = k_normed

        # BOS is decoded at position 0, so RoPE is identity.  llama.cpp then
        # writes normalized K and raw V to the default F16 KV cache before
        # attention reads them back.  With one key softmax is exactly [1], so
        # only the cached V value contributes to the attention output, but we
        # round-trip K too because this is the state representation required by
        # subsequent multi-token decoding.
        k_cache = f16_roundtrip(k_normed)
        v_cache = f16_roundtrip(v)
        pregate = gqa_one_key_attention(v_cache)
        checkpoints["attn_pregate-3"] = pregate
        gate_sigmoid = [base.sigmoid(x) for x in gate]
        checkpoints["gate_sigmoid-3"] = gate_sigmoid
        gated = [pregate[i] * gate_sigmoid[i] for i in range(Q_DIM)]
        checkpoints["attn_gated-3"] = gated

        attn_out = runtime.matvec(
            view("attn_output.weight"), metas["blk.3.attn_output.weight"], gated
        )
        checkpoints["attn_output-3"] = attn_out
        residual = [hidden[i] + attn_out[i] for i in range(HIDDEN)]
        checkpoints["attn_residual-3"] = residual
        post_norm = base.rms_norm(residual, vec("post_attention_norm.weight"))
        checkpoints["attn_post_norm-3"] = post_norm

        prepared = runtime.quantize(post_norm, "Q6_K")
        ffn_gate = runtime.matvec(
            view("ffn_gate.weight"), metas["blk.3.ffn_gate.weight"], post_norm, prepared
        )
        ffn_up = runtime.matvec(
            view("ffn_up.weight"), metas["blk.3.ffn_up.weight"], post_norm, prepared
        )
        swiglu = [base.silu(ffn_gate[i]) * ffn_up[i] for i in range(INTERMEDIATE)]
        ffn_out = runtime.matvec(
            view("ffn_down.weight"), metas["blk.3.ffn_down.weight"], swiglu
        )
        checkpoints["ffn_out-3"] = ffn_out
        final = [residual[i] + ffn_out[i] for i in range(HIDDEN)]
        checkpoints["post_ffn-3"] = final
        reader_report = reader.report()
        bound.release()

    comparisons: dict[str, Any] = {}
    missing: list[str] = []
    for name, cand in checkpoints.items():
        ref = reference.get(name)
        if ref is None:
            missing.append(name)
        else:
            comparisons[name] = metrics(ref, cand)

    thresholds = {
        "layer3_input": (2e-6, 2e-6),
        "attn_norm-3": (2e-5, 2e-5),
        "Qcur_full-3": (5e-4, 2e-4),
        "Qcur_reshaped-3": (5e-4, 2e-4),
        "Qcur_normed-3": (8e-4, 4e-4),
        "Kcur_normed-3": (8e-4, 4e-4),
        "gate_reshaped-3": (5e-4, 2e-4),
        "Vcur-3": (5e-4, 2e-4),
        "attn_pregate-3": (8e-4, 4e-4),
        "gate_sigmoid-3": (5e-4, 2e-4),
        "attn_gated-3": (1e-3, 5e-4),
        "attn_output-3": (3e-3, 1e-3),
        "attn_residual-3": (3e-3, 1e-3),
        "attn_post_norm-3": (3e-3, 1e-3),
        "ffn_out-3": (6e-3, 2e-3),
        "post_ffn-3": (6e-3, 2e-3),
    }
    failures: list[str] = []
    for name, (max_lim, rel_lim) in thresholds.items():
        if name in missing:
            failures.append(f"{name}: missing oracle checkpoint")
            continue
        m = comparisons[name]
        if m["max_abs"] > max_lim or m["relative_l2"] > rel_lim:
            failures.append(
                f"{name}: max_abs={m['max_abs']:.6g}>{max_lim:g} or "
                f"rel_l2={m['relative_l2']:.6g}>{rel_lim:g}"
            )

    result = {
        "schema": "qwen38-k3-real-q6-layer3-attn-semantic-v2",
        "status": "PASS" if not failures else "FAIL",
        "failure_class": None if not failures else "model correctness",
        "model_sha256": base.SHA256,
        "official_revision": base.REVISION,
        "llama_cpp_reference_revision": base.LLAMA_CPP_REVISION,
        "token_id": int(oracle["token_id"]),
        "position": 0,
        "input_source": oracle.get("layer3_input_source"),
        "one_key_attention_softmax_exact": True,
        "gqa_repeat": GQA_REPEAT,
        "kv_cache": {
            "type_k": "F16",
            "type_v": "F16",
            "k_roundtrip_applied": True,
            "v_roundtrip_applied": True,
            "k_values": len(k_cache),
            "v_values": len(v_cache),
        },
        "oracle_checkpoint_notes": {
            "Qcur_reshaped-3": "canonicalized from Qcur_full-3 interleaved Q/G logical layout",
            "Kcur-3": "pinned optimized oracle aliases raw K with Kcur_normed; diagnostic-only and not gated",
        },
        "candidate": {
            "storage": "existing K3Trunk one-slot layer3",
            "native_quant_matvec": True,
            "full_matrix_dequantized": False,
            "activation_quantizations": runtime.activation_quantizations,
            "matvec_rows": runtime.matvec_rows,
            "reader_report": reader_report,
        },
        "comparisons": comparisons,
        "thresholds": {k: {"max_abs": v[0], "relative_l2": v[1]} for k, v in thresholds.items()},
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
    qg: list[float] = []
    for h in range(N_HEAD):
        qg.extend([float(h)] * HEAD_DIM)
        qg.extend([1000.0 + float(h)] * HEAD_DIM)
    q, gate = split_q_gate(qg)
    if q[0] != 0.0 or q[HEAD_DIM] != 1.0 or gate[0] != 1000.0 or gate[HEAD_DIM] != 1001.0:
        raise SystemExit("interleaved Q/G split sanity failed")

    f16_probe = f16_roundtrip([1.0 / 3.0, -1.0 / 3.0])
    if f16_probe != [0.333251953125, -0.333251953125]:
        raise SystemExit(f"F16 KV-cache round-trip sanity failed: {f16_probe}")

    v = flatten([[float(h) + 1.0 / 3.0] * HEAD_DIM for h in range(N_HEAD_KV)])
    v_cache = f16_roundtrip(v)
    pregate = gqa_one_key_attention(v_cache)
    heads = split_heads(pregate, N_HEAD)
    expected = [f16_roundtrip([float(h // GQA_REPEAT) + 1.0 / 3.0])[0] for h in range(N_HEAD)]
    if [head[0] for head in heads] != expected:
        raise SystemExit("GQA one-key cached-V repeat sanity failed")

    normed = rms_norm_heads([0.1] * Q_DIM, N_HEAD, [1.0] * HEAD_DIM)
    if len(normed) != Q_DIM or not all(math.isfinite(x) for x in normed):
        raise SystemExit("per-head RMSNorm sanity failed")

    print(json.dumps({
        "schema": "qwen38-k3-layer3-attn-zero-model-sanity-v2",
        "status": "PASS",
        "n_head": N_HEAD,
        "n_head_kv": N_HEAD_KV,
        "head_dim": HEAD_DIM,
        "qg_interleaved_per_head": True,
        "gqa_repeat": GQA_REPEAT,
        "position0_rope_identity": True,
        "one_key_softmax": 1.0,
        "kv_cache_type_k": "F16",
        "kv_cache_type_v": "F16",
        "invalid_raw_k_oracle_is_not_required": True,
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
        raise SystemExit("GGUF gate currently requires a little-endian host")
    if args.cmd == "sanity":
        sanity()
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        execute(args.model, args.native_lib, args.inventory, args.oracle, args.work_dir, args.output)


if __name__ == "__main__":
    main()
