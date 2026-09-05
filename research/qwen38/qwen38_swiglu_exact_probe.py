#!/usr/bin/env python3
"""Real bitwise A/B gate for exact native Qwen3.8 SwiGLU.

Both sides use the current proven staged-prefill path:
  * exact multi-vector quantized matvec;
  * exact native F32 matvec;
  * exact native full-attention core;
  * exact native GDN causal-conv + SiLU;
  * exact native GDN output RMSNorm + SiLU gate.

The only candidate change is replacing the Python elementwise FFN SwiGLU loop
with one exact C call per decoder layer.  Hidden vectors, recurrent state,
attention KV state, known SHA anchors, and K3 traffic must remain identical.
"""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_attention_core_exact_probe as attn_probe
import qwen38_gdn_output_gate_exact_probe as gate_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from attention_core_runtime import ExactAttentionCore
from gdn_conv_silu_runtime import ExactGDNConvSilu
from gdn_output_gate_runtime import ExactGDNOutputGate
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many
from swiglu_runtime import ExactSwiGLU

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def _python_swiglu_reference(
    gate_rows: Sequence[Sequence[float]],
    up_rows: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [
            gen.mulf(gen.t2.siluf(gate_rows[j][i]), up_rows[j][i])
            for i in range(len(gate_rows[j]))
        ]
        for j in range(len(gate_rows))
    ]


def sanity(swiglu_lib: Path) -> None:
    gen.exact.install()
    core = ExactSwiGLU(swiglu_lib)
    for rows, width in ((1, 1), (1, 3), (3, 17), (3, 128), (2, gdn.INTERMEDIATE)):
        gate = [
            [
                gen.f32((((j * width + i) * 37 + width * 11) % 2047 - 1023) / 31.0)
                for i in range(width)
            ]
            for j in range(rows)
        ]
        up = [
            [
                gen.f32((((j * width + i) * 53 + 7) % 1601 - 800) / 29.0)
                for i in range(width)
            ]
            for j in range(rows)
        ]
        ref = _python_swiglu_reference(gate, up)
        cand = core.compute(gate, up)
        if len(ref) != len(cand) or any(
            _f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref, cand)
        ):
            raise SystemExit(f"native SwiGLU bitwise mismatch rows={rows} width={width}")
    print("QWEN38_SWIGLU_EXACT_SANITY PASS")


def _run_with_patches(
    engine,
    *,
    attn_core: ExactAttentionCore,
    conv_core: ExactGDNConvSilu,
    gate_core: ExactGDNOutputGate,
    swiglu_core: ExactSwiGLU | None,
):
    stats = {
        "native_attention_core_seconds": 0.0,
        "native_conv_silu_seconds": 0.0,
        "native_output_gate_seconds": 0.0,
        "native_swiglu_seconds": 0.0,
    }
    old_attn = prefill._full_attention_layer_many
    old_rec = prefill._recurrent_layer_many
    old_ffn = prefill._ffn_many

    def native_attn(engine_, hidden, il, pos0, view, metas, vec):
        local = {"native_attention_core_seconds": stats["native_attention_core_seconds"]}
        out = attn_probe._full_attention_layer_native(
            engine_, hidden, il, pos0, view, metas, vec, attn_core, local)
        stats["native_attention_core_seconds"] = local["native_attention_core_seconds"]
        return out

    def native_rec(engine_, hidden, il, view, metas, vec):
        local = {
            "native_conv_silu_seconds": stats["native_conv_silu_seconds"],
            "native_output_gate_seconds": stats["native_output_gate_seconds"],
        }
        out = gate_probe._recurrent_layer_native_conv_gate(
            conv_core, gate_core, local, engine_, hidden, il, view, metas, vec)
        stats["native_conv_silu_seconds"] = local["native_conv_silu_seconds"]
        stats["native_output_gate_seconds"] = local["native_output_gate_seconds"]
        return out

    def native_ffn(runtime, view, metas, prefix, xs):
        prepared = runtime.prepare_many(xs, "Q6_K")
        gate = runtime.matvec_many(
            view("ffn_gate.weight"), metas[f"{prefix}.ffn_gate.weight"], xs,
            prepared=prepared)
        up = runtime.matvec_many(
            view("ffn_up.weight"), metas[f"{prefix}.ffn_up.weight"], xs,
            prepared=prepared)
        t0 = time.monotonic()
        sw = swiglu_core.compute(gate, up)
        stats["native_swiglu_seconds"] += time.monotonic() - t0
        return runtime.matvec_many(
            view("ffn_down.weight"), metas[f"{prefix}.ffn_down.weight"], sw)

    prefill._full_attention_layer_many = native_attn
    prefill._recurrent_layer_many = native_rec
    if swiglu_core is not None:
        prefill._ffn_many = native_ffn
    try:
        before = int(engine.reader.report()["bytes_read"])
        t0 = time.monotonic()
        hidden = prefill.step_block_many(engine, PROMPT_IDS)
        seconds = time.monotonic() - t0
        k3_bytes = int(engine.reader.report()["bytes_read"]) - before
        return hidden, seconds, k3_bytes, stats
    finally:
        prefill._ffn_many = old_ffn
        prefill._recurrent_layer_many = old_rec
        prefill._full_attention_layer_many = old_attn


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        initial = pair.capture_state(engine)

        ref_attn = ExactAttentionCore(args.attn_lib)
        ref_conv = ExactGDNConvSilu(args.conv_lib)
        ref_gate = ExactGDNOutputGate(args.gate_lib)
        ref_hidden, ref_seconds, ref_bytes, ref_stats = _run_with_patches(
            engine,
            attn_core=ref_attn,
            conv_core=ref_conv,
            gate_core=ref_gate,
            swiglu_core=None,
        )
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        cand_attn = ExactAttentionCore(args.attn_lib)
        cand_conv = ExactGDNConvSilu(args.conv_lib)
        cand_gate = ExactGDNOutputGate(args.gate_lib)
        swiglu_core = ExactSwiGLU(args.swiglu_lib)
        cand_hidden, cand_seconds, cand_bytes, cand_stats = _run_with_patches(
            engine,
            attn_core=cand_attn,
            conv_core=cand_conv,
            gate_core=cand_gate,
            swiglu_core=swiglu_core,
        )
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("native SwiGLU candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native SwiGLU candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("native SwiGLU A/B requires direct I/O")

        swiglu_report = swiglu_core.report()
        expected_calls = 64
        expected_rows = expected_calls * len(PROMPT_IDS)
        expected_values = expected_rows * gdn.INTERMEDIATE
        if (
            swiglu_report["calls"] != expected_calls
            or swiglu_report["rows"] != expected_rows
            or swiglu_report["values"] != expected_values
        ):
            raise RuntimeError(f"unexpected native SwiGLU coverage: {swiglu_report}")

        payload = {
            "schema": "qwen38-swiglu-exact-ab-v1",
            "status": "PASS",
            "claim": "exact native FFN SwiGLU; same-run real GGUF A/B",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "hidden_sha256": cand_hidden_sha,
            "state_sha256": cand_state_sha,
            "reference_seconds_same_run": ref_seconds,
            "candidate_seconds_same_run": cand_seconds,
            "speedup_same_run": ref_seconds / cand_seconds if cand_seconds else None,
            "reference_k3_bytes": ref_bytes,
            "candidate_k3_bytes": cand_bytes,
            "reference_native_seconds": ref_stats,
            "candidate_native_seconds": cand_stats,
            "native_swiglu": swiglu_report,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": engine.reader.report(),
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "research_basis": [
                "Liger treats SwiGLU as an operation-fusion target to eliminate Python/framework intermediate work.",
                "Deep Kernel Fusion for Transformers identifies SwiGLU MLP cache and intermediate traffic as a major inference optimization target.",
                "This kernel changes no reduction order: each element preserves the pinned F32 expf, sigmoid, SiLU, and multiply sequence.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_SWIGLU_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--swiglu-lib", type=Path, required=True)
    r = sub.add_parser("real")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--quant-lib", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--attn-lib", type=Path, required=True)
    r.add_argument("--conv-lib", type=Path, required=True)
    r.add_argument("--gate-lib", type=Path, required=True)
    r.add_argument("--swiglu-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.swiglu_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
