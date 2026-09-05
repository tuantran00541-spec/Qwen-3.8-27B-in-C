#!/usr/bin/env python3
"""Real same-run A/B gate for exact native Qwen3.8 RMSNorm.

Reference and candidate both use the current best exact staged-prefill path:
quantized matvec-many, native F32 matvec, native attention core, native GDN
conv+SiLU, native GDN output gate, and native FFN SwiGLU.  The candidate only
replaces the pinned Python ggml RMSNorm helpers with the serial exact C kernel.
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
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
import qwen38_swiglu_exact_probe as sw_probe
from attention_core_runtime import ExactAttentionCore
from gdn_conv_silu_runtime import ExactGDNConvSilu
from gdn_output_gate_runtime import ExactGDNOutputGate
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many
from rmsnorm_runtime import ExactRMSNorm
from swiglu_runtime import ExactSwiGLU

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256

EXPECTED_VECTOR_CALLS = 1408
EXPECTED_VECTOR_ROWS = 1408
EXPECTED_VECTOR_VALUES = 1408 * gdn.HIDDEN
EXPECTED_HEAD_CALLS = 16 * len(PROMPT_IDS) * 2
EXPECTED_HEAD_ROWS = 16 * len(PROMPT_IDS) * (gen.attn.N_HEAD + gen.attn.N_HEAD_KV)
EXPECTED_HEAD_VALUES = EXPECTED_HEAD_ROWS * gen.attn.HEAD_DIM


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity(rmsnorm_lib: Path) -> None:
    gen.exact.install()
    core = ExactRMSNorm(rmsnorm_lib)
    for width in (1, 2, 3, 17, 128, gen.attn.HEAD_DIM, gdn.HIDDEN):
        x = [
            gen.f32(((i * 37 + width * 11) % 2047 - 1023) / 31.0)
            for i in range(width)
        ]
        w = [
            gen.f32(((i * 19 + 97) % 701 + 200) / 997.0)
            for i in range(width)
        ]
        ref = gen.rmswrap.ggml_rms_norm(x, w, gdn.RMS_EPS)
        cand = core.compute(x, w, gdn.RMS_EPS)
        if _f32_bytes(ref) != _f32_bytes(cand):
            raise SystemExit(f"native RMSNorm bitwise mismatch width={width}")

    for heads in (1, gen.attn.N_HEAD_KV, gen.attn.N_HEAD):
        width = gen.attn.HEAD_DIM
        values = [
            gen.f32(((i * 43 + heads * 13) % 4093 - 2046) / 47.0)
            for i in range(heads * width)
        ]
        weight = [gen.f32(((i * 23 + 17) % 503 + 300) / 887.0) for i in range(width)]
        ref = gen.rmswrap.ggml_rms_norm_heads(values, heads, weight)
        cand = core.compute_heads(values, heads, weight, gen.attn.RMS_EPS)
        if _f32_bytes(ref) != _f32_bytes(cand):
            raise SystemExit(f"native RMSNorm-heads bitwise mismatch heads={heads}")
    print("QWEN38_RMSNORM_EXACT_SANITY PASS")


def _best_run(engine, args, *, rms_core: ExactRMSNorm | None):
    old_rms = gdn.rms_norm
    old_heads = gen.attn.rms_norm_heads
    rms_seconds = 0.0

    if rms_core is not None:
        def native_rms(values, weight, eps=gdn.RMS_EPS):
            nonlocal rms_seconds
            t0 = time.monotonic()
            out = rms_core.compute(values, weight, eps)
            rms_seconds += time.monotonic() - t0
            return out

        def native_heads(values, heads, weight):
            nonlocal rms_seconds
            t0 = time.monotonic()
            out = rms_core.compute_heads(values, heads, weight, gen.attn.RMS_EPS)
            rms_seconds += time.monotonic() - t0
            return out

        gdn.rms_norm = native_rms
        gen.attn.rms_norm_heads = native_heads

    attn_core = ExactAttentionCore(args.attn_lib)
    conv_core = ExactGDNConvSilu(args.conv_lib)
    gate_core = ExactGDNOutputGate(args.gate_lib)
    swiglu_core = ExactSwiGLU(args.swiglu_lib)
    try:
        hidden, seconds, k3_bytes, stats = sw_probe._run_with_patches(
            engine,
            attn_core=attn_core,
            conv_core=conv_core,
            gate_core=gate_core,
            swiglu_core=swiglu_core,
        )
        stats = dict(stats)
        stats["native_rmsnorm_seconds"] = rms_seconds
        return hidden, seconds, k3_bytes, stats
    finally:
        gdn.rms_norm = old_rms
        gen.attn.rms_norm_heads = old_heads


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        initial = pair.capture_state(engine)

        ref_hidden, ref_seconds, ref_bytes, ref_stats = _best_run(engine, args, rms_core=None)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        rms_core = ExactRMSNorm(args.rmsnorm_lib)
        cand_hidden, cand_seconds, cand_bytes, cand_stats = _best_run(
            engine, args, rms_core=rms_core)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("native RMSNorm candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native RMSNorm candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("native RMSNorm A/B requires direct I/O")

        report = rms_core.report()
        expected = {
            "calls": EXPECTED_VECTOR_CALLS,
            "rows": EXPECTED_VECTOR_ROWS,
            "values": EXPECTED_VECTOR_VALUES,
            "head_calls": EXPECTED_HEAD_CALLS,
            "head_rows": EXPECTED_HEAD_ROWS,
            "head_values": EXPECTED_HEAD_VALUES,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise RuntimeError(f"unexpected native RMSNorm coverage {key}: {report}")

        payload = {
            "schema": "qwen38-rmsnorm-exact-ab-v1",
            "status": "PASS",
            "claim": "exact native serial RMSNorm; same-run real GGUF A/B",
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
            "native_rmsnorm": report,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "profile_basis": {
                "cprofile_run": 33822343545,
                "cprofile_artifact": 9918865983,
                "python_f32_calls_observed": 97693336,
                "python_ggml_rms_norm_calls_observed": 6336,
                "note": "cProfile timings motivated the target; same-run A/B here is the speed evidence.",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_RMSNORM_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--rmsnorm-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib", "inventory", "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.rmsnorm_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
