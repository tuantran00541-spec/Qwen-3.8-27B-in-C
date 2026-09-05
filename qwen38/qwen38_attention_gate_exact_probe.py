#!/usr/bin/env python3
"""Exact native full-attention sigmoid-gate A/B on the current-best Qwen3.8 path.

Reference and candidate both keep Q8 allocation-hoist, native RMSNorm, native
SwiGLU, native residual additions, native attention core, and native GDN
helpers. The candidate only replaces the Python sigmoid_f32 + mulf loop after
full-attention with one exact C call per token/layer.
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
import qwen38_residual_add_exact_probe as res_probe
from attention_gate_runtime import ExactAttentionGate
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many
from residual_add_runtime import ExactResidualAdd

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_GATE_CALLS = 16 * len(PROMPT_IDS)
EXPECTED_GATE_VALUES = EXPECTED_GATE_CALLS * gen.attn.Q_DIM
EXPECTED_RESIDUAL = {"calls": 128, "rows": 1408, "values": 7208960}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity(gate_lib: Path) -> None:
    gen.exact.install()
    core = ExactAttentionGate(gate_lib)
    for n in (1, 3, 17, 128, gen.attn.Q_DIM):
        pregate = [
            gen.f32(((i * 37 + 11) % 4093 - 2046) / 31.0)
            for i in range(n)
        ]
        gate = [
            gen.f32(((i * 53 + 7) % 3079 - 1539) / 29.0)
            for i in range(n)
        ]
        ref = [
            gen.mulf(pregate[i], gen.exact.sigmoid_f32(gate[i]))
            for i in range(n)
        ]
        cand = core.compute(pregate, gate)
        if _f32_bytes(ref) != _f32_bytes(cand):
            raise SystemExit(f"native attention-gate bitwise mismatch n={n}")
    print("QWEN38_ATTENTION_GATE_EXACT_SANITY PASS")


def _full_attention_with_native_gate(
    engine, hidden, il: int, pos0: int, view, metas, vec,
    core, stats: dict[str, float],
    residual_core: ExactResidualAdd, residual_stats: dict[str, float],
    attention_gate_core: ExactAttentionGate,
    attention_gate_stats: dict[str, float],
):
    runtime = engine.runtime
    p = f"blk.{il}"
    attn_norm_w = vec("attn_norm.weight")
    xs = [gdn.rms_norm(h, attn_norm_w) for h in hidden]

    qg_rows = runtime.matvec_many(view("attn_q.weight"), metas[f"{p}.attn_q.weight"], xs)
    k_rows = runtime.matvec_many(view("attn_k.weight"), metas[f"{p}.attn_k.weight"], xs)
    v_rows = runtime.matvec_many(view("attn_v.weight"), metas[f"{p}.attn_v.weight"], xs)
    q_norm_w = vec("attn_q_norm.weight")
    k_norm_w = vec("attn_k_norm.weight")
    cache = engine.caches[il]
    gated_rows: list[list[float]] = []

    for j in range(len(hidden)):
        q, gate = gen.attn.split_q_gate(qg_rows[j])
        q = gen.attn.rms_norm_heads(q, gen.attn.N_HEAD, q_norm_w)
        k = gen.attn.rms_norm_heads(k_rows[j], gen.attn.N_HEAD_KV, k_norm_w)
        pos = pos0 + j
        q_rope = gen.t2.rope_text_neox(q, gen.attn.N_HEAD, pos)
        k_rope = gen.t2.rope_text_neox(k, gen.attn.N_HEAD_KV, pos)
        cache["k"].append(gen.attn.f16_roundtrip(k_rope))
        cache["v"].append(gen.attn.f16_roundtrip(v_rows[j]))

        t0 = time.monotonic()
        pregate = core.compute(
            il,
            q_rope,
            cache,
            q_heads=gen.attn.N_HEAD,
            kv_heads=gen.attn.N_HEAD_KV,
            head_dim=gen.attn.HEAD_DIM,
            scale=gen.t2.SCALE_ATTN,
        )
        stats["native_attention_core_seconds"] += time.monotonic() - t0

        t0 = time.monotonic()
        gated_rows.append(attention_gate_core.compute(pregate, gate))
        attention_gate_stats["native_attention_gate_seconds"] += time.monotonic() - t0

    ao = runtime.matvec_many(
        view("attn_output.weight"), metas[f"{p}.attn_output.weight"], gated_rows)
    t0 = time.monotonic()
    residual = residual_core.compute(hidden, ao)
    residual_stats["native_residual_add_seconds"] += time.monotonic() - t0
    post_norm_w = vec("post_attention_norm.weight")
    post = [gdn.rms_norm(r, post_norm_w) for r in residual]
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    t0 = time.monotonic()
    out = residual_core.compute(residual, fo)
    residual_stats["native_residual_add_seconds"] += time.monotonic() - t0
    return out


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        initial = pair.capture_state(engine)

        ref_residual = ExactResidualAdd(args.residual_lib)
        ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms = res_probe._best_run(
            engine, args, residual_core=ref_residual)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        cand_residual = ExactResidualAdd(args.residual_lib)
        attention_gate_core = ExactAttentionGate(args.attention_gate_lib)
        attention_gate_stats = {"native_attention_gate_seconds": 0.0}
        old_full_attention = res_probe._full_attention_with_residual

        def candidate_full_attention(
            engine_, hidden, il, pos0, view, metas, vec, core, stats,
            residual_core, residual_stats,
        ):
            return _full_attention_with_native_gate(
                engine_, hidden, il, pos0, view, metas, vec, core, stats,
                residual_core, residual_stats,
                attention_gate_core, attention_gate_stats,
            )

        res_probe._full_attention_with_residual = candidate_full_attention
        try:
            cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms = res_probe._best_run(
                engine, args, residual_core=cand_residual)
        finally:
            res_probe._full_attention_with_residual = old_full_attention

        cand_stats = dict(cand_stats)
        cand_stats.update(attention_gate_stats)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("native attention-gate candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native attention-gate candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("native attention-gate A/B requires direct I/O")
        if ref_residual.report() != EXPECTED_RESIDUAL or cand_residual.report() != EXPECTED_RESIDUAL:
            raise RuntimeError(
                f"unexpected residual coverage ref={ref_residual.report()} cand={cand_residual.report()}")
        if ref_rms.get("total_rows") != 6336 or cand_rms.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected RMSNorm coverage ref={ref_rms} cand={cand_rms}")

        gate_report = attention_gate_core.report()
        if gate_report["calls"] != EXPECTED_GATE_CALLS:
            raise RuntimeError(
                f"attention-gate calls {gate_report['calls']} != {EXPECTED_GATE_CALLS}")
        if gate_report["values"] != EXPECTED_GATE_VALUES:
            raise RuntimeError(
                f"attention-gate values {gate_report['values']} != {EXPECTED_GATE_VALUES}")

        payload = {
            "schema": "qwen38-attention-gate-exact-ab-v1",
            "status": "PASS",
            "claim": "exact native full-attention sigmoid gate; same-run real GGUF A/B on Q8-noalloc current-best path",
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
            "reference_residual": ref_residual.report(),
            "candidate_residual": cand_residual.report(),
            "reference_rmsnorm": ref_rms,
            "candidate_rmsnorm": cand_rms,
            "attention_gate": gate_report,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "python_scalar_values_removed": EXPECTED_GATE_VALUES,
                "reference": "Python sigmoid_f32 + mulf per attention output value",
                "candidate": "one exact native C sigmoid-gate call per token/full-attention layer",
                "arithmetic_change": False,
            },
            "baseline_q8_noalloc_ab_run": 33836777802,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_ATTENTION_GATE_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--attention-gate-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib", "residual-lib",
        "attention-gate-lib", "inventory", "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.attention_gate_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
