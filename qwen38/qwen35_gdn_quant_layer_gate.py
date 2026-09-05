#!/usr/bin/env python3
"""Real Qwen3.8-27B Q6_K_L layer-0 semantic gate for the custom K3 runtime.

The gate has two phases:

* ``inventory`` verifies the pinned GGUF and emits a machine-readable decoder
  contract before any risky semantic work.
* ``run`` packs only decoder layer 0 into the existing K3 trunk, executes the
  layer with the existing native Q6_K/Q8_0 scalar matvec bridge, and compares
  semantic checkpoints against a pinned llama.cpp ``cb_eval`` oracle running
  on the exact same GGUF/token.

This is deliberately correctness-first. It does not dequantize whole matrices
and it does not introduce another scheduler or tensor lookup path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
from typing import Any, Sequence

from gguf_k3_layout import pack_gguf_layers, partition_tensors
from gguf_quant_ref import dequantize_q8_0, row_nbytes
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

REPO = "bartowski/Qwen3.8-27B-GGUF"
FILE = "Qwen3.8-27B-Q6_K_L.gguf"
SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LLAMA_CPP_REVISION = "557614e0296ff4a5b6f649737a65ae2076eea2fd"

DECODER_LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
K_HEADS = 16
V_HEADS = 48
HEAD_DIM = 128
KEY_DIM = K_HEADS * HEAD_DIM
VALUE_DIM = V_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CONV_KERNEL = 4
RMS_EPS = 1e-6
VOCAB = 248320

RECURRENT_SUFFIXES = {
    "attn_gate.weight": ("Q6_K", [HIDDEN, VALUE_DIM]),
    "attn_norm.weight": ("F32", [HIDDEN]),
    "attn_qkv.weight": ("Q8_0", [HIDDEN, CONV_DIM]),
    "ffn_down.weight": ("Q6_K", [INTERMEDIATE, HIDDEN]),
    "ffn_gate.weight": ("Q6_K", [HIDDEN, INTERMEDIATE]),
    "ffn_up.weight": ("Q6_K", [HIDDEN, INTERMEDIATE]),
    "post_attention_norm.weight": ("F32", [HIDDEN]),
    "ssm_a": ("F32", [V_HEADS]),
    "ssm_alpha.weight": ("F32", [HIDDEN, V_HEADS]),
    "ssm_beta.weight": ("F32", [HIDDEN, V_HEADS]),
    "ssm_conv1d.weight": ("F32", [CONV_KERNEL, CONV_DIM]),
    "ssm_dt.bias": ("F32", [V_HEADS]),
    "ssm_norm.weight": ("F32", [HEAD_DIM]),
    "ssm_out.weight": ("Q8_0", [VALUE_DIM, HIDDEN]),
}
FULL_SUFFIXES = {
    "attn_k.weight": ("Q8_0", [HIDDEN, 1024]),
    "attn_k_norm.weight": ("F32", [256]),
    "attn_norm.weight": ("F32", [HIDDEN]),
    "attn_output.weight": ("Q8_0", [VALUE_DIM, HIDDEN]),
    "attn_q.weight": ("Q8_0", [HIDDEN, 12288]),
    "attn_q_norm.weight": ("F32", [256]),
    "attn_v.weight": ("Q8_0", [HIDDEN, 1024]),
    "ffn_down.weight": ("Q6_K", [INTERMEDIATE, HIDDEN]),
    "ffn_gate.weight": ("Q6_K", [HIDDEN, INTERMEDIATE]),
    "ffn_up.weight": ("Q6_K", [HIDDEN, INTERMEDIATE]),
    "post_attention_norm.weight": ("F32", [HIDDEN]),
}
EXPECTED_GLOBALS = {
    "token_embd.weight": ("Q8_0", [HIDDEN, VOCAB]),
    "output_norm.weight": ("F32", [HIDDEN]),
    "output.weight": ("Q8_0", [HIDDEN, VOCAB]),
}
EXPECTED_Q4_MTP = {
    "blk.64.attn_k.weight",
    "blk.64.attn_output.weight",
    "blk.64.attn_q.weight",
    "blk.64.attn_v.weight",
    "blk.64.ffn_down.weight",
    "blk.64.ffn_gate.weight",
    "blk.64.ffn_up.weight",
    "blk.64.nextn.eh_proj.weight",
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_tensor(t, expected_type: str, expected_shape: Sequence[int], errors: list[str]) -> None:
    if t.type_name != expected_type:
        errors.append(f"{t.name}: type={t.type_name} expected={expected_type}")
    if list(t.shape) != list(expected_shape):
        errors.append(f"{t.name}: shape={list(t.shape)} expected={list(expected_shape)}")


def inventory(model: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    state: dict[str, Any] = {
        "schema": "qwen38-gguf-decoder-contract-v1",
        "status": "INCOMPLETE",
        "repo": REPO,
        "file": FILE,
        "expected_sha256": SHA256,
        "file_bytes": model.stat().st_size,
    }
    atomic_json(output, state)
    digest = sha256_file(model)
    state.update(sha256=digest, sha256_match=(digest == SHA256))
    atomic_json(output, state)
    if digest != SHA256:
        state.update(status="FAIL", failure_class="parser/layout", error="SHA256 mismatch")
        atomic_json(output, state)
        raise RuntimeError(f"GGUF SHA256 mismatch: {digest}")

    directory = parse_gguf(model)
    grouped, auxiliary, globals_ = partition_tensors(directory, expected_layers=DECODER_LAYERS)
    errors: list[str] = []
    if directory.metadata.get("general.architecture") != "qwen35":
        errors.append(f"architecture={directory.metadata.get('general.architecture')!r}")
    if sorted(grouped) != list(range(DECODER_LAYERS)):
        errors.append("decoder block set is not exactly 0..63")
    if sorted(auxiliary) != [64]:
        errors.append(f"auxiliary blocks={sorted(auxiliary)} expected=[64]")

    decoder_types: Counter[str] = Counter()
    layer_contract: list[dict[str, Any]] = []
    for layer in range(DECODER_LAYERS):
        expected = FULL_SUFFIXES if layer % 4 == 3 else RECURRENT_SUFFIXES
        actual = {t.name.removeprefix(f"blk.{layer}."): t for t in grouped[layer]}
        if set(actual) != set(expected):
            errors.append(
                f"blk.{layer}: names differ missing={sorted(set(expected)-set(actual))} "
                f"extra={sorted(set(actual)-set(expected))}"
            )
        for suffix, (typ, shape) in expected.items():
            if suffix in actual:
                validate_tensor(actual[suffix], typ, shape, errors)
        decoder_types.update(t.type_name for t in grouped[layer])
        layer_contract.append({
            "layer": layer,
            "kind": "full_attention" if layer % 4 == 3 else "gated_deltanet",
            "tensor_count": len(grouped[layer]),
            "types": dict(sorted(Counter(t.type_name for t in grouped[layer]).items())),
        })

    global_map = {t.name: t for t in globals_}
    if set(global_map) != set(EXPECTED_GLOBALS):
        errors.append(f"global tensors={sorted(global_map)} expected={sorted(EXPECTED_GLOBALS)}")
    for name, (typ, shape) in EXPECTED_GLOBALS.items():
        if name in global_map:
            validate_tensor(global_map[name], typ, shape, errors)

    q4_names = {t.name for t in directory.tensors if t.type_name == "Q4_0"}
    if q4_names != EXPECTED_Q4_MTP:
        errors.append(f"Q4_0 names={sorted(q4_names)} expected={sorted(EXPECTED_Q4_MTP)}")
    if decoder_types.get("Q4_0", 0):
        errors.append("decoder main path unexpectedly contains Q4_0")
    unsupported_decoder = set(decoder_types) - {"F32", "Q6_K", "Q8_0"}
    if unsupported_decoder:
        errors.append(f"unsupported decoder types={sorted(unsupported_decoder)}")

    state.update({
        "status": "PASS" if not errors else "FAIL",
        "failure_class": None if not errors else "parser/layout",
        "architecture": directory.metadata.get("general.architecture"),
        "tensor_count": directory.tensor_count,
        "tensor_type_counts": dict(sorted(Counter(t.type_name for t in directory.tensors).items())),
        "decoder_type_counts": dict(sorted(decoder_types.items())),
        "decoder_layer_count": len(grouped),
        "auxiliary_block_ids": sorted(auxiliary),
        "global_tensors": [
            {"name": t.name, "type": t.type_name, "shape": list(t.shape), "nbytes": t.nbytes}
            for t in globals_
        ],
        "q4_0_names": sorted(q4_names),
        "layer_contract": layer_contract,
        "layer0_tensors": [
            {"name": t.name, "type": t.type_name, "shape": list(t.shape), "nbytes": t.nbytes}
            for t in grouped[0]
        ],
        "layer3_tensors": [
            {"name": t.name, "type": t.type_name, "shape": list(t.shape), "nbytes": t.nbytes}
            for t in grouped[3]
        ],
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": rss_gib(),
    })
    atomic_json(output, state)
    if errors:
        raise RuntimeError("decoder contract failed: " + "; ".join(errors[:8]))
    return state


def _load_native(path: Path):
    lib = ctypes.CDLL(str(path))
    c_u8p = ctypes.POINTER(ctypes.c_uint8)
    c_fp = ctypes.POINTER(ctypes.c_float)
    lib.qwen_quantize_q8_k_scalar.argtypes = [c_fp, ctypes.c_size_t, c_u8p, ctypes.c_size_t]
    lib.qwen_quantize_q8_k_scalar.restype = ctypes.c_int
    lib.qwen_quantize_q8_0_scalar.argtypes = [c_fp, ctypes.c_size_t, c_u8p, ctypes.c_size_t]
    lib.qwen_quantize_q8_0_scalar.restype = ctypes.c_int
    for name in ("qwen_matvec_q6_k_q8_k_scalar", "qwen_matvec_q8_0_q8_0_scalar"):
        fn = getattr(lib, name)
        fn.argtypes = [c_u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                       c_u8p, ctypes.c_size_t, c_fp]
        fn.restype = ctypes.c_int
    return lib


class QuantRuntime:
    def __init__(self, lib):
        self.lib = lib
        self.activation_quantizations = 0
        self.matvec_rows = 0

    def quantize(self, x: Sequence[float], kind: str):
        n = len(x)
        x_arr = (ctypes.c_float * n)(*map(float, x))
        if kind == "Q6_K":
            if n % 256:
                raise ValueError("Q6_K activation width must be divisible by 256")
            nbytes = (n // 256) * 292
            fn = self.lib.qwen_quantize_q8_k_scalar
        elif kind == "Q8_0":
            if n % 32:
                raise ValueError("Q8_0 activation width must be divisible by 32")
            nbytes = (n // 32) * 34
            fn = self.lib.qwen_quantize_q8_0_scalar
        else:
            raise ValueError(kind)
        buf = (ctypes.c_uint8 * nbytes)()
        rc = fn(x_arr, n, buf, nbytes)
        if rc != 0:
            raise RuntimeError(f"activation quantization {kind} failed rc={rc}")
        self.activation_quantizations += 1
        return buf, nbytes

    def matvec(self, weights: memoryview, meta: dict[str, Any], x: Sequence[float], prepared=None) -> list[float]:
        kind = meta["type_name"]
        ne0, rows = map(int, meta["shape"])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: x={len(x)} ne0={ne0}")
        if kind == "F32":
            return f32_matvec(weights, ne0, rows, x)
        if kind not in {"Q6_K", "Q8_0"}:
            raise ValueError(f"unsupported matvec type {kind}")
        if prepared is None:
            prepared = self.quantize(x, kind)
        activation, activation_bytes = prepared
        w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
        out = (ctypes.c_float * rows)()
        fn = (self.lib.qwen_matvec_q6_k_q8_k_scalar if kind == "Q6_K"
              else self.lib.qwen_matvec_q8_0_q8_0_scalar)
        rc = fn(w_arr, len(weights), rows, ne0, activation, activation_bytes, out)
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: native matvec failed rc={rc}")
        self.matvec_rows += rows
        return [float(out[i]) for i in range(rows)]


def f32_vector(view: memoryview) -> list[float]:
    if len(view) % 4:
        raise ValueError("F32 tensor byte count is not divisible by four")
    return [x[0] for x in struct.iter_unpack("<f", view)]


def f32_matvec(view: memoryview, ne0: int, rows: int, x: Sequence[float]) -> list[float]:
    vals = f32_vector(view)
    if len(vals) != ne0 * rows:
        raise ValueError("F32 matrix size mismatch")
    out: list[float] = []
    for row in range(rows):
        base = row * ne0
        out.append(math.fsum(vals[base + i] * float(x[i]) for i in range(ne0)))
    return out


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = RMS_EPS) -> list[float]:
    if len(x) != len(weight):
        raise ValueError("RMSNorm shape mismatch")
    inv = 1.0 / math.sqrt(math.fsum(float(v) * float(v) for v in x) / len(x) + eps)
    return [float(v) * inv * float(w) for v, w in zip(x, weight)]


def l2_norm(x: Sequence[float], eps: float = RMS_EPS) -> list[float]:
    # Match pinned ggml CPU L2_NORM exactly at the semantic level: eps is a
    # floor on ||x||_2, not an additive term inside the square root.
    denom = max(math.sqrt(math.fsum(float(v) * float(v) for v in x)), eps)
    return [float(v) / denom for v in x]


def split_heads(x: Sequence[float], heads: int) -> list[list[float]]:
    if len(x) != heads * HEAD_DIM:
        raise ValueError("head reshape mismatch")
    return [list(map(float, x[h * HEAD_DIM:(h + 1) * HEAD_DIM])) for h in range(heads)]


def flatten(heads: Sequence[Sequence[float]]) -> list[float]:
    return [float(v) for head in heads for v in head]


def one_token_core(q: Sequence[float], k: Sequence[float], v: Sequence[float], beta: Sequence[float]) -> list[float]:
    """Initial-state one-token DeltaNet in GGUF tiled head order."""
    qh = split_heads(q, K_HEADS)
    kh = split_heads(k, K_HEADS)
    vh = split_heads(v, V_HEADS)
    qn = [l2_norm(h) for h in qh]
    kn = [l2_norm(h) for h in kh]
    repeats = V_HEADS // K_HEADS
    qrep = [h for _ in range(repeats) for h in qn]
    krep = [h for _ in range(repeats) for h in kn]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out: list[list[float]] = []
    for h in range(V_HEADS):
        score = math.fsum(qrep[h][i] * krep[h][i] for i in range(HEAD_DIM)) * scale
        coeff = float(beta[h]) * score
        out.append([coeff * value for value in vh[h]])
    return flatten(out)


def metrics(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    if len(ref) != len(cand):
        return {"length_mismatch": float(abs(len(ref) - len(cand))), "max_abs": math.inf,
                "rmse": math.inf, "relative_l2": math.inf}
    diffs = [float(b) - float(a) for a, b in zip(ref, cand)]
    max_abs = max((abs(v) for v in diffs), default=0.0)
    mse = math.fsum(v * v for v in diffs) / max(1, len(diffs))
    ref2 = math.fsum(float(v) * float(v) for v in ref)
    err2 = math.fsum(v * v for v in diffs)
    return {
        "max_abs": max_abs,
        "rmse": math.sqrt(mse),
        "relative_l2": math.sqrt(err2 / ref2) if ref2 else math.sqrt(err2),
    }


def _layer_meta(manifest: dict[str, Any], layer: int = 0) -> dict[str, dict[str, Any]]:
    entry = next(x for x in manifest["layers"] if int(x["layer"]) == layer)
    return {t["name"]: t for t in entry["tensors"]}


def _embedding_row(model: Path, directory, token_id: int) -> list[float]:
    tensor = directory.by_name()["token_embd.weight"]
    ne0, ne1 = map(int, tensor.shape)
    if tensor.type_name != "Q8_0" or ne0 != HIDDEN or not 0 <= token_id < ne1:
        raise ValueError("unexpected token embedding contract")
    stride = row_nbytes("Q8_0", ne0)
    fd = os.open(model, os.O_RDONLY)
    try:
        raw = os.pread(fd, stride, tensor.data_offset + token_id * stride)
    finally:
        os.close(fd)
    if len(raw) != stride:
        raise EOFError("short token embedding row")
    return dequantize_q8_0(raw, ne0)


def execute_layer0(model: Path, native_lib: Path, inventory_json: Path, oracle_json: Path,
                   work_dir: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    inv = json.loads(inventory_json.read_text(encoding="utf-8"))
    if inv.get("status") != "PASS" or inv.get("sha256") != SHA256:
        raise RuntimeError("inventory contract is not a PASS for the pinned GGUF")
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != "qwen38-llama-layer0-oracle-v1" or not oracle.get("captured_complete_layer"):
        raise RuntimeError("llama layer0 oracle is incomplete")
    token_id = int(oracle["token_id"])

    directory = parse_gguf(model)
    trunk = work_dir / "layer0.k3.bin"
    manifest_path = work_dir / "layer0.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, manifest_path, layers=[0], model_id=MODEL_ID, revision=REVISION,
        source_sha256=SHA256, expected_layers=DECODER_LAYERS,
    )
    layer_entry = manifest["layers"][0]
    budget = int(layer_entry["read_bytes"])
    lib = _load_native(native_lib)
    runtime = QuantRuntime(lib)
    checkpoints: dict[str, list[float]] = {}

    hidden = _embedding_row(model, directory, token_id)
    checkpoints["model.input_embed"] = hidden

    with K3Trunk(trunk, manifest_path, budget_bytes=budget, want_ring=1,
                 max_pinned=0, prefer_direct_io=True) as reader:
        layer = reader.bind(0)
        metas = _layer_meta(manifest, 0)

        def view(name: str) -> memoryview:
            return reader.tensor_view(layer, name)

        def vec(name: str) -> list[float]:
            return f32_vector(view(name))

        attn_norm = rms_norm(hidden, vec("blk.0.attn_norm.weight"))
        checkpoints["attn_norm-0"] = attn_norm

        qkv = runtime.matvec(view("blk.0.attn_qkv.weight"), metas["blk.0.attn_qkv.weight"], attn_norm)
        z = runtime.matvec(view("blk.0.attn_gate.weight"), metas["blk.0.attn_gate.weight"], attn_norm)
        beta_raw = runtime.matvec(view("blk.0.ssm_beta.weight"), metas["blk.0.ssm_beta.weight"], attn_norm)
        alpha = runtime.matvec(view("blk.0.ssm_alpha.weight"), metas["blk.0.ssm_alpha.weight"], attn_norm)
        checkpoints["linear_attn_qkv_mixed-0"] = qkv
        checkpoints["z-0"] = z
        checkpoints["beta-0"] = beta_raw
        checkpoints["alpha-0"] = alpha

        beta = [sigmoid(v) for v in beta_raw]
        checkpoints["beta_sigmoid-0"] = beta
        dt = vec("blk.0.ssm_dt.bias")
        a = vec("blk.0.ssm_a")
        a_softplus = [softplus(alpha[i] + dt[i]) for i in range(V_HEADS)]
        gate = [a[i] * a_softplus[i] for i in range(V_HEADS)]
        checkpoints["a_softplus-0"] = a_softplus
        checkpoints["gate-0"] = gate

        kernels = vec("blk.0.ssm_conv1d.weight")
        if len(kernels) != CONV_DIM * CONV_KERNEL:
            raise ValueError("conv kernel shape mismatch")
        conv = [silu(qkv[c] * kernels[c * CONV_KERNEL + (CONV_KERNEL - 1)]) for c in range(CONV_DIM)]
        checkpoints["conv_output_silu-0"] = conv
        q = conv[:KEY_DIM]
        k = conv[KEY_DIM:2 * KEY_DIM]
        v = conv[2 * KEY_DIM:]

        qn = [l2_norm(h) for h in split_heads(q, K_HEADS)]
        kn = [l2_norm(h) for h in split_heads(k, K_HEADS)]
        repeats = V_HEADS // K_HEADS
        q_predelta = flatten([h for _ in range(repeats) for h in qn])
        k_predelta = flatten([h for _ in range(repeats) for h in kn])
        checkpoints["q_conv_predelta-0"] = q_predelta
        checkpoints["k_conv_predelta-0"] = k_predelta
        checkpoints["v_conv_predelta-0"] = v

        core = one_token_core(q, k, v, beta)
        norm_w = vec("blk.0.ssm_norm.weight")
        core_heads = split_heads(core, V_HEADS)
        z_heads = split_heads(z, V_HEADS)
        gated: list[list[float]] = []
        for ch, zh in zip(core_heads, z_heads):
            inv_rms = 1.0 / math.sqrt(math.fsum(x * x for x in ch) / HEAD_DIM + RMS_EPS)
            gated.append([ch[d] * inv_rms * norm_w[d] * silu(zh[d]) for d in range(HEAD_DIM)])
        gated_flat = flatten(gated)

        linear_out = runtime.matvec(view("blk.0.ssm_out.weight"), metas["blk.0.ssm_out.weight"], gated_flat)
        checkpoints["linear_attn_out-0"] = linear_out
        residual = [hidden[i] + linear_out[i] for i in range(HIDDEN)]
        checkpoints["attn_residual-0"] = residual
        post_norm = rms_norm(residual, vec("blk.0.post_attention_norm.weight"))
        checkpoints["attn_post_norm-0"] = post_norm

        prepared_mlp = runtime.quantize(post_norm, "Q6_K")
        gate_mlp = runtime.matvec(view("blk.0.ffn_gate.weight"), metas["blk.0.ffn_gate.weight"], post_norm, prepared_mlp)
        up_mlp = runtime.matvec(view("blk.0.ffn_up.weight"), metas["blk.0.ffn_up.weight"], post_norm, prepared_mlp)
        swiglu = [silu(gate_mlp[i]) * up_mlp[i] for i in range(INTERMEDIATE)]
        ffn_out = runtime.matvec(view("blk.0.ffn_down.weight"), metas["blk.0.ffn_down.weight"], swiglu)
        checkpoints["ffn_out-0"] = ffn_out
        final = [residual[i] + ffn_out[i] for i in range(HIDDEN)]
        checkpoints["post_ffn-0"] = final
        reader_report = reader.report()
        layer.release()

    reference = oracle["checkpoints"]
    comparisons: dict[str, Any] = {}
    missing = []
    for name, cand in checkpoints.items():
        ref = reference.get(name)
        if ref is None:
            missing.append(name)
            continue
        comparisons[name] = metrics(ref, cand)

    required = set(checkpoints)
    pass_thresholds = {
        "model.input_embed": (2e-6, 2e-6),
        "attn_norm-0": (2e-5, 2e-5),
        "linear_attn_qkv_mixed-0": (5e-4, 2e-4),
        "z-0": (5e-4, 2e-4),
        "beta-0": (2e-5, 2e-5),
        "alpha-0": (2e-5, 2e-5),
        "beta_sigmoid-0": (2e-5, 2e-5),
        "a_softplus-0": (2e-5, 2e-5),
        "gate-0": (2e-5, 2e-5),
        "conv_output_silu-0": (8e-4, 4e-4),
        "q_conv_predelta-0": (5e-4, 3e-4),
        "k_conv_predelta-0": (5e-4, 3e-4),
        "v_conv_predelta-0": (8e-4, 4e-4),
        "linear_attn_out-0": (3e-3, 1e-3),
        "attn_residual-0": (3e-3, 1e-3),
        "attn_post_norm-0": (3e-3, 1e-3),
        "ffn_out-0": (6e-3, 2e-3),
        "post_ffn-0": (6e-3, 2e-3),
    }
    failed: list[str] = []
    for name in sorted(required):
        if name in missing:
            failed.append(f"{name}: missing oracle checkpoint")
            continue
        max_lim, rel_lim = pass_thresholds[name]
        m = comparisons[name]
        if m["max_abs"] > max_lim or m["relative_l2"] > rel_lim:
            failed.append(
                f"{name}: max_abs={m['max_abs']:.6g}>{max_lim:g} or rel_l2={m['relative_l2']:.6g}>{rel_lim:g}"
            )

    result = {
        "schema": "qwen38-k3-real-q6-layer0-semantic-v1",
        "status": "PASS" if not failed else "FAIL",
        "failure_class": None if not failed else "model correctness",
        "model_sha256": SHA256,
        "official_revision": REVISION,
        "llama_cpp_reference_revision": LLAMA_CPP_REVISION,
        "token_id": token_id,
        "initial_recurrent_state": "zero",
        "initial_conv_history": "zero",
        "candidate": {
            "storage": "existing K3Trunk one-slot layer0",
            "native_quant_matvec": True,
            "full_matrix_dequantized": False,
            "activation_quantizations": runtime.activation_quantizations,
            "matvec_rows": runtime.matvec_rows,
            "reader_report": reader_report,
        },
        "comparisons": comparisons,
        "thresholds": {k: {"max_abs": v[0], "relative_l2": v[1]} for k, v in pass_thresholds.items()},
        "failures": failed,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": rss_gib(),
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)
    return result


def sanity() -> None:
    q = [0.01 * ((i % 17) - 8) for i in range(KEY_DIM)]
    k = [0.01 * ((i % 19) - 9) for i in range(KEY_DIM)]
    v = [0.005 * ((i % 23) - 11) for i in range(VALUE_DIM)]
    beta = [0.2 + 0.01 * (i % 7) for i in range(V_HEADS)]
    out = one_token_core(q, k, v, beta)
    if len(out) != VALUE_DIM or not all(math.isfinite(x) for x in out):
        raise SystemExit("one-token core sanity failed")

    # Pinned ggml CPU L2_NORM uses 1/fmax(sqrt(sum(x^2)), eps).
    # The first case catches the historical additive-epsilon bug; the second
    # locks the actual floor behavior when the norm is below eps.
    l2_regular = l2_norm([3e-4, 4e-4])
    if max(abs(a - b) for a, b in zip(l2_regular, [0.6, 0.8])) > 1e-12:
        raise SystemExit(f"ggml L2 norm regular-case sanity failed: {l2_regular}")
    l2_floor = l2_norm([3e-7, 4e-7])
    if max(abs(a - b) for a, b in zip(l2_floor, [0.3, 0.4])) > 1e-12:
        raise SystemExit(f"ggml L2 norm floor-case sanity failed: {l2_floor}")

    a = [-math.exp(-1.0 + i * 0.01) for i in range(V_HEADS)]
    alpha = [0.1 * math.sin(i) for i in range(V_HEADS)]
    dt = [0.01 * math.cos(i) for i in range(V_HEADS)]
    gate = [a[i] * softplus(alpha[i] + dt[i]) for i in range(V_HEADS)]
    if not all(x < 0.0 for x in gate):
        raise SystemExit("converted ssm_a gate sanity failed")
    print(json.dumps({
        "schema": "qwen38-k3-layer0-zero-model-sanity-v2",
        "status": "PASS",
        "value_dim": VALUE_DIM,
        "tiled_repeats": V_HEADS // K_HEADS,
        "ggml_l2_norm_epsilon_semantics": "floor_on_l2_norm",
        "converter_ssm_a_is_pretransformed_negative_exp": True,
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    p_inv = sub.add_parser("inventory")
    p_inv.add_argument("--model", type=Path, required=True)
    p_inv.add_argument("--output", type=Path, required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--model", type=Path, required=True)
    p_run.add_argument("--native-lib", type=Path, required=True)
    p_run.add_argument("--inventory", type=Path, required=True)
    p_run.add_argument("--oracle", type=Path, required=True)
    p_run.add_argument("--work-dir", type=Path, required=True)
    p_run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if sys.byteorder != "little":
        raise SystemExit("GGUF gate currently requires a little-endian host")
    if args.cmd == "sanity":
        sanity()
    elif args.cmd == "inventory":
        inventory(args.model, args.output)
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        execute_layer0(args.model, args.native_lib, args.inventory, args.oracle, args.work_dir, args.output)


if __name__ == "__main__":
    main()
