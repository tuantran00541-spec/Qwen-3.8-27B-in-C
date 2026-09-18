#!/usr/bin/env python3
"""Bonsai 2 layer-0 semantic gate against the pinned Prism llama.cpp oracle.

This is the first complete recurrent-layer gate for the custom native runtime:
PTQ1/PQ2 low-bit weights stay encoded in the K3 ring, activations receive the
Prism sign+Hadamard fold, BF16 alpha/beta use the native BF16 kernel, token
embeddings are inverse-transformed after row lookup, and ssm_out applies the
Qwen3.5 tiled->grouped V-head permutation before the fold.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
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

MODEL_ID = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
MODEL_SHA256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
PRISM_REVISION = "d8f26eec7"
LOWBIT = {"PTQ1_0": (128, 28), "PQ2_0": (128, 34)}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def metrics(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    if len(ref) != len(cand):
        return {"max_abs": math.inf, "rmse": math.inf, "relative_l2": math.inf}
    diffs = [float(b) - float(a) for a, b in zip(ref, cand)]
    err2 = math.fsum(v * v for v in diffs)
    ref2 = math.fsum(float(v) * float(v) for v in ref)
    return {
        "max_abs": max((abs(v) for v in diffs), default=0.0),
        "rmse": math.sqrt(err2 / max(1, len(diffs))),
        "relative_l2": math.sqrt(err2 / ref2) if ref2 else math.sqrt(err2),
    }


def embedding_row(
    model: Path,
    directory,
    runtime: Bonsai2NativeRuntime,
    token_id: int,
) -> list[float]:
    tensor = directory.by_name()["token_embd.weight"]
    ne0, rows = map(int, tensor.shape)
    if tensor.type_name not in LOWBIT or ne0 != gdn.HIDDEN or not 0 <= token_id < rows:
        raise ValueError(
            f"unexpected Bonsai token embedding contract type={tensor.type_name} "
            f"shape={list(tensor.shape)} token={token_id}"
        )
    qk, block = LOWBIT[tensor.type_name]
    stride = (ne0 // qk) * block
    if stride * rows != tensor.nbytes:
        raise ValueError("Bonsai token embedding row geometry mismatch")
    fd = os.open(model, os.O_RDONLY)
    try:
        raw = os.pread(fd, stride, tensor.data_offset + token_id * stride)
    finally:
        os.close(fd)
    if len(raw) != stride:
        raise EOFError("short Bonsai token embedding row")
    return runtime.dequantize_lookup_row(raw, tensor.type_name, ne0, tensor.name)


def run(
    model: Path,
    native_lib: Path,
    oracle_path: Path,
    work_dir: Path,
    output: Path,
    threads: int,
) -> dict[str, Any]:
    started = time.monotonic()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if not oracle.get("captured_complete_layer"):
        raise RuntimeError(f"Prism layer0 oracle incomplete: {oracle.get('error')}")
    token_id = int(oracle["token_id"])
    reference = oracle["checkpoints"]

    directory = parse_gguf(model)
    if directory.metadata.get("general.architecture") != "qwen35":
        raise ValueError("Bonsai 2 GGUF architecture is not qwen35")
    runtime = Bonsai2NativeRuntime(
        native_lib, directory.metadata, threads=threads, max_rows=gdn.INTERMEDIATE
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    trunk = work_dir / "layer0.k3.bin"
    index = work_dir / "layer0.k3.json"
    manifest = pack_gguf_layers(
        directory,
        trunk,
        index,
        layers=[0],
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        source_sha256=MODEL_SHA256,
        expected_layers=gdn.DECODER_LAYERS,
    )
    entry = manifest["layers"][0]
    metas = {t["name"]: t for t in entry["tensors"]}
    checkpoints: dict[str, list[float]] = {}

    hidden = embedding_row(model, directory, runtime, token_id)
    checkpoints["model.input_embed"] = hidden

    with K3Trunk(
        trunk,
        index,
        budget_bytes=int(entry["read_bytes"]),
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as reader:
        bound = reader.bind(0)

        def view(suffix: str) -> memoryview:
            return reader.tensor_view(bound, f"blk.0.{suffix}")

        def meta(suffix: str):
            return metas[f"blk.0.{suffix}"]

        def vec(suffix: str) -> list[float]:
            return gdn.f32_vector(view(suffix))

        attn_norm = gdn.rms_norm(hidden, vec("attn_norm.weight"))
        checkpoints["attn_norm-0"] = attn_norm

        # qkv + z share the same folded 5120-wide activation.
        prepared_attn = runtime.prepare_activation("blk.0.attn_qkv.weight", attn_norm)
        qkv = runtime.matvec_prepared(view("attn_qkv.weight"), meta("attn_qkv.weight"), prepared_attn)
        z = runtime.matvec_prepared(view("attn_gate.weight"), meta("attn_gate.weight"), prepared_attn)
        beta_raw = runtime.matvec(view("ssm_beta.weight"), meta("ssm_beta.weight"), attn_norm)
        alpha = runtime.matvec(view("ssm_alpha.weight"), meta("ssm_alpha.weight"), attn_norm)
        checkpoints["linear_attn_qkv_mixed-0"] = qkv
        checkpoints["z-0"] = z
        checkpoints["beta-0"] = beta_raw
        checkpoints["alpha-0"] = alpha

        beta = [gdn.sigmoid(v) for v in beta_raw]
        checkpoints["beta_sigmoid-0"] = beta
        dt = vec("ssm_dt.bias")
        aa = vec("ssm_a")
        a_softplus = [gdn.softplus(alpha[i] + dt[i]) for i in range(gdn.V_HEADS)]
        gate = [aa[i] * a_softplus[i] for i in range(gdn.V_HEADS)]
        checkpoints["a_softplus-0"] = a_softplus
        checkpoints["gate-0"] = gate

        kernels = vec("ssm_conv1d.weight")
        if len(kernels) != gdn.CONV_DIM * gdn.CONV_KERNEL:
            raise ValueError("conv kernel shape mismatch")
        conv = [
            gdn.silu(qkv[c] * kernels[c * gdn.CONV_KERNEL + (gdn.CONV_KERNEL - 1)])
            for c in range(gdn.CONV_DIM)
        ]
        checkpoints["conv_output_silu-0"] = conv
        q = conv[: gdn.KEY_DIM]
        k = conv[gdn.KEY_DIM : 2 * gdn.KEY_DIM]
        v = conv[2 * gdn.KEY_DIM :]

        qn = [gdn.l2_norm(h) for h in gdn.split_heads(q, gdn.K_HEADS)]
        kn = [gdn.l2_norm(h) for h in gdn.split_heads(k, gdn.K_HEADS)]
        repeats = gdn.V_HEADS // gdn.K_HEADS
        q_predelta = gdn.flatten([h for _ in range(repeats) for h in qn])
        k_predelta = gdn.flatten([h for _ in range(repeats) for h in kn])
        checkpoints["q_conv_predelta-0"] = q_predelta
        checkpoints["k_conv_predelta-0"] = k_predelta
        checkpoints["v_conv_predelta-0"] = v

        core = gdn.one_token_core(q, k, v, beta)
        norm_w = vec("ssm_norm.weight")
        core_heads = gdn.split_heads(core, gdn.V_HEADS)
        z_heads = gdn.split_heads(z, gdn.V_HEADS)
        gated: list[list[float]] = []
        for ch, zh in zip(core_heads, z_heads):
            inv_rms = 1.0 / math.sqrt(
                math.fsum(x * x for x in ch) / gdn.HEAD_DIM + gdn.RMS_EPS
            )
            gated.append([
                ch[d] * inv_rms * norm_w[d] * gdn.silu(zh[d])
                for d in range(gdn.HEAD_DIM)
            ])
        gated_flat = gdn.flatten(gated)

        # Runtime performs Prism's Qwen3.5 tiled->grouped V-head permutation
        # before explicit signs + blockwise normalized Hadamard.
        linear_out = runtime.matvec(
            view("ssm_out.weight"), meta("ssm_out.weight"), gated_flat
        )
        checkpoints["linear_attn_out-0"] = linear_out
        residual = [hidden[i] + linear_out[i] for i in range(gdn.HIDDEN)]
        checkpoints["attn_residual-0"] = residual
        post_norm = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
        checkpoints["attn_post_norm-0"] = post_norm

        prepared_ffn = runtime.prepare_activation("blk.0.ffn_gate.weight", post_norm)
        gate_mlp = runtime.matvec_prepared(
            view("ffn_gate.weight"), meta("ffn_gate.weight"), prepared_ffn
        )
        up_mlp = runtime.matvec_prepared(
            view("ffn_up.weight"), meta("ffn_up.weight"), prepared_ffn
        )
        swiglu = [
            gdn.silu(gate_mlp[i]) * up_mlp[i] for i in range(gdn.INTERMEDIATE)
        ]
        ffn_out = runtime.matvec(view("ffn_down.weight"), meta("ffn_down.weight"), swiglu)
        checkpoints["ffn_out-0"] = ffn_out
        final = [residual[i] + ffn_out[i] for i in range(gdn.HIDDEN)]
        checkpoints["post_ffn-0"] = final
        reader_report = reader.report()

        bound.release()

    comparisons = {
        name: metrics(reference[name], cand)
        for name, cand in checkpoints.items()
        if name in reference
    }
    missing = sorted(set(checkpoints) - set(reference))

    # The oracle and candidate use the exact same PTQ1 model. Thresholds allow
    # normal CPU reduction-order differences while remaining tight enough to
    # catch a wrong Hadamard/sign/permutation boundary immediately.
    thresholds = {
        "model.input_embed": (3e-5, 1e-5),
        "attn_norm-0": (8e-5, 3e-5),
        "linear_attn_qkv_mixed-0": (2e-3, 8e-4),
        "z-0": (2e-3, 8e-4),
        "beta-0": (3e-4, 1e-4),
        "alpha-0": (3e-4, 1e-4),
        "beta_sigmoid-0": (2e-4, 1e-4),
        "a_softplus-0": (3e-4, 1e-4),
        "gate-0": (3e-4, 1e-4),
        "conv_output_silu-0": (3e-3, 1e-3),
        "q_conv_predelta-0": (2e-3, 8e-4),
        "k_conv_predelta-0": (2e-3, 8e-4),
        "v_conv_predelta-0": (3e-3, 1e-3),
        "linear_attn_out-0": (8e-3, 3e-3),
        "attn_residual-0": (8e-3, 3e-3),
        "attn_post_norm-0": (8e-3, 3e-3),
        "ffn_out-0": (1.5e-2, 5e-3),
        "post_ffn-0": (1.5e-2, 5e-3),
    }
    failures: list[str] = []
    # Prism/ggml may optimize these simple recurrent intermediates away before
    # the callback sees them. Their effects are still covered by the later
    # linear_attn_out/residual checkpoints, so absence is not a parity failure.
    optional_oracle = {"beta_sigmoid-0", "a_softplus-0", "gate-0"}
    for name in checkpoints:
        if name in missing:
            if name not in optional_oracle:
                failures.append(f"{name}: missing oracle checkpoint")
            continue
        m = comparisons[name]
        max_lim, rel_lim = thresholds[name]
        if m["max_abs"] > max_lim or m["relative_l2"] > rel_lim:
            failures.append(
                f"{name}: max_abs={m['max_abs']:.6g}>{max_lim:g} "
                f"or relative_l2={m['relative_l2']:.6g}>{rel_lim:g}"
            )

    state = {
        "schema": "qwen38-bonsai2-layer0-semantic-v1",
        "status": "PASS" if not failures else "FAIL",
        "model_sha256": MODEL_SHA256,
        "model_revision": MODEL_REVISION,
        "prism_revision": PRISM_REVISION,
        "token_id": token_id,
        "threads": threads,
        "comparisons": comparisons,
        "thresholds": {
            k: {"max_abs": v[0], "relative_l2": v[1]}
            for k, v in thresholds.items()
        },
        "failures": failures,
        "optional_oracle_checkpoints": sorted(optional_oracle),
        "runtime": runtime.report(),
        "reader": reader_report,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": rss_gib(),
    }
    runtime.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    print("QWEN38_BONSAI2_LAYER0_SEMANTIC_PASS")
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--native-lib", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    run(
        args.model,
        args.native_lib,
        args.oracle,
        args.work_dir,
        args.output,
        args.threads,
    )


if __name__ == "__main__":
    main()
