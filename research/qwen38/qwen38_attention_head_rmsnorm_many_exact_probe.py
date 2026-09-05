#!/usr/bin/env python3
"""Exact full-attention Q/K head RMSNorm batching A/B on current-best Qwen3.8.

Reference includes the proven normal full-attention RMSNorm-many path. Candidate
changes only the Python/ctypes boundary for Q/K per-head RMSNorm: each layer's
11 Q rows and 11 K rows are marshaled in two calls while the C bridge invokes
the already-proven exact per-token head RMSNorm entry point strictly row by
row. No head arithmetic or reduction order changes.
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
import qwen38_attention_rmsnorm_many_exact_probe as attn_rms_probe
import qwen38_gdn_rmsnorm_many_exact_probe as gdn_rms_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from fast_quantize_runtime import FastActivationQuantizer, find_base_quant_runtime
from gdn_repeat_scale_runtime import ExactGDNRepeatScale
from gdn_state_batch_runtime import ExactGDNStateBatch
from native_f32_runtime import enable_native_f32
from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime
from quant_many_runtime import load_many_lib
from rmsnorm_heads_many_runtime import ExactRMSNormHeadsMany
from rmsnorm_many_runtime import ExactRMSNormMany
from rmsnorm_runtime import ExactRMSNorm

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_MANY = dict(gdn_rms_probe.EXPECTED_MANY)
EXPECTED_REPEAT = dict(gdn_rms_probe.EXPECTED_REPEAT)
EXPECTED_BATCH = dict(gdn_rms_probe.EXPECTED_BATCH)
EXPECTED_FAST_QUANT_CALLS = gdn_rms_probe.EXPECTED_FAST_QUANT_CALLS
EXPECTED_GDN_RMS_MANY = dict(gdn_rms_probe.EXPECTED_RMS_MANY)
EXPECTED_ATTN_RMS_MANY = dict(attn_rms_probe.EXPECTED_ATTN_RMS_MANY)
EXPECTED_HEAD_MANY = {
    "calls": 16 * 2,
    "token_rows": 16 * len(PROMPT_IDS) * 2,
    "head_rows": 16 * len(PROMPT_IDS) * (gen.attn.N_HEAD + gen.attn.N_HEAD_KV),
    "values": 16 * len(PROMPT_IDS) * (gen.attn.Q_DIM + gen.attn.KV_DIM),
}
EXPECTED_REF_WRAPPER = {
    "calls": 0,
    "rows": 0,
    "values": 0,
    "head_calls": 16 * len(PROMPT_IDS) * 2,
    "head_rows": EXPECTED_HEAD_MANY["head_rows"],
    "head_values": EXPECTED_HEAD_MANY["values"],
    "total_rows": EXPECTED_HEAD_MANY["head_rows"],
    "total_values": EXPECTED_HEAD_MANY["values"],
}
EXPECTED_ZERO_WRAPPER = {
    "calls": 0,
    "rows": 0,
    "values": 0,
    "head_calls": 0,
    "head_rows": 0,
    "head_values": 0,
    "total_rows": 0,
    "total_values": 0,
}
EXPECTED_TOTAL_RMS_ROWS = 6336
EXPECTED_TOTAL_RMS_VALUES = 8470528
EXPECTED_READER = dict(attn_rms_probe.EXPECTED_READER)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def _fixture(rows: int, width: int, salt: int) -> list[list[float]]:
    return [
        [
            (((r * width + i) * 104729 + salt * 1543) % 200003 - 100001)
            / 4096.0
            for i in range(width)
        ]
        for r in range(rows)
    ]


def sanity(rmsnorm_lib: Path, head_many_lib: Path) -> None:
    scalar = ExactRMSNorm(rmsnorm_lib)
    many = ExactRMSNormHeadsMany(head_many_lib)
    rows = len(PROMPT_IDS)
    weight = [
        (((i * 8191 + 37) % 32749) - 16374) / 8192.0
        for i in range(gen.attn.HEAD_DIM)
    ]
    cases = (
        ("q", gen.attn.N_HEAD, gen.attn.Q_DIM, 17),
        ("k", gen.attn.N_HEAD_KV, gen.attn.KV_DIM, 31),
    )
    for site, heads, width, salt in cases:
        values = _fixture(rows, width, salt)
        ref = [scalar.compute_heads(row, heads, weight, gdn.RMS_EPS) for row in values]
        cand = many.compute(values, heads, weight, gdn.RMS_EPS)
        if len(ref) != len(cand) or any(
            _f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref, cand)
        ):
            raise SystemExit(f"head RMSNorm-many bitwise mismatch site={site}")
    expected = {
        "calls": 2,
        "token_rows": rows * 2,
        "head_rows": rows * (gen.attn.N_HEAD + gen.attn.N_HEAD_KV),
        "values": rows * (gen.attn.Q_DIM + gen.attn.KV_DIM),
    }
    if many.report() != expected:
        raise SystemExit(f"unexpected head RMSNorm-many sanity coverage: {many.report()}")
    print("QWEN38_ATTENTION_HEAD_RMSNORM_MANY_EXACT_SANITY PASS")


def _full_attention_with_head_rms_many(
    engine, hidden, il: int, pos0: int, view, metas, vec,
    core, stats: dict[str, float],
    residual_core, residual_stats: dict[str, float],
    attention_gate_core, attention_gate_stats: dict[str, float],
    rms_many: ExactRMSNormMany,
    rms_many_stats: dict[str, float],
    head_many: ExactRMSNormHeadsMany,
    head_many_stats: dict[str, float],
):
    runtime = engine.runtime
    p = f"blk.{il}"
    attn_norm_w = vec("attn_norm.weight")
    t0 = time.monotonic()
    xs = rms_many.compute(hidden, attn_norm_w, gdn.RMS_EPS)
    rms_many_stats["native_attention_rmsnorm_many_seconds"] += time.monotonic() - t0

    qg_rows = runtime.matvec_many(view("attn_q.weight"), metas[f"{p}.attn_q.weight"], xs)
    k_rows = runtime.matvec_many(view("attn_k.weight"), metas[f"{p}.attn_k.weight"], xs)
    v_rows = runtime.matvec_many(view("attn_v.weight"), metas[f"{p}.attn_v.weight"], xs)
    q_norm_w = vec("attn_q_norm.weight")
    k_norm_w = vec("attn_k_norm.weight")

    q_rows: list[list[float]] = []
    gate_rows: list[list[float]] = []
    for qg in qg_rows:
        q, gate = gen.attn.split_q_gate(qg)
        q_rows.append(q)
        gate_rows.append(gate)

    t0 = time.monotonic()
    q_norm_rows = head_many.compute(
        q_rows, gen.attn.N_HEAD, q_norm_w, gdn.RMS_EPS)
    k_norm_rows = head_many.compute(
        k_rows, gen.attn.N_HEAD_KV, k_norm_w, gdn.RMS_EPS)
    head_many_stats["native_attention_head_rmsnorm_many_seconds"] += time.monotonic() - t0

    cache = engine.caches[il]
    gated_rows: list[list[float]] = []
    for j in range(len(hidden)):
        q = q_norm_rows[j]
        k = k_norm_rows[j]
        gate = gate_rows[j]
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
    t0 = time.monotonic()
    post = rms_many.compute(residual, post_norm_w, gdn.RMS_EPS)
    rms_many_stats["native_attention_rmsnorm_many_seconds"] += time.monotonic() - t0
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    t0 = time.monotonic()
    out = residual_core.compute(residual, fo)
    residual_stats["native_residual_add_seconds"] += time.monotonic() - t0
    return out


def _run_with_head_rms_many(
    engine,
    args,
    repeat_core: ExactGDNRepeatScale,
    state_batch_core: ExactGDNStateBatch,
    gdn_rms_many: ExactRMSNormMany,
    attention_rms_many: ExactRMSNormMany,
    head_many: ExactRMSNormHeadsMany,
):
    old = attn_rms_probe._full_attention_with_normal_rms_many
    head_many_stats = {"native_attention_head_rmsnorm_many_seconds": 0.0}

    def impl(
        engine_, hidden, il, pos0, view, metas, vec, core, stats,
        residual_core, residual_stats, attention_gate_core, attention_gate_stats,
        rms_many, rms_many_stats,
    ):
        return _full_attention_with_head_rms_many(
            engine_, hidden, il, pos0, view, metas, vec, core, stats,
            residual_core, residual_stats, attention_gate_core, attention_gate_stats,
            rms_many, rms_many_stats, head_many, head_many_stats,
        )

    attn_rms_probe._full_attention_with_normal_rms_many = impl
    try:
        result = attn_rms_probe._run_with_attention_rms_many(
            engine, args, repeat_core, state_batch_core,
            gdn_rms_many, attention_rms_many)
    finally:
        attn_rms_probe._full_attention_with_normal_rms_many = old
    hidden, seconds, k3_bytes, stats, rms, residual, attn_gate, repeat_report = result
    stats = dict(stats)
    stats.update(head_many_stats)
    return hidden, seconds, k3_bytes, stats, rms, residual, attn_gate, repeat_report


def _combined_rms_rows(wrapper, gdn_many, attn_many, head_many=None) -> int:
    return (
        int(wrapper["total_rows"])
        + int(gdn_many["rows"])
        + int(attn_many["rows"])
        + (0 if head_many is None else int(head_many["head_rows"]))
    )


def _combined_rms_values(wrapper, gdn_many, attn_many, head_many=None) -> int:
    return (
        int(wrapper["total_values"])
        + int(gdn_many["values"])
        + int(attn_many["values"])
        + (0 if head_many is None else int(head_many["values"]))
    )


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    ref_fast: FastActivationQuantizer | None = None
    cand_fast: FastActivationQuantizer | None = None
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        base_runtime = engine.runtime
        base_quant = find_base_quant_runtime(base_runtime)
        initial = pair.capture_state(engine)

        ref_fast = FastActivationQuantizer(base_quant)
        ref_fast.install()
        ref_many = FastMarshalQuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = ref_many
        ref_repeat = ExactGDNRepeatScale(args.repeat_lib)
        ref_state_batch = ExactGDNStateBatch(args.batch_state_lib)
        ref_gdn_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        ref_attention_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        (
            ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms,
            ref_residual, ref_attn_gate, ref_repeat_report,
        ) = attn_rms_probe._run_with_attention_rms_many(
            engine, args, ref_repeat, ref_state_batch,
            ref_gdn_rms_many, ref_attention_rms_many)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)
        ref_many_report = ref_many.report()
        ref_fast_report = ref_fast.report()
        ref_state_batch_report = ref_state_batch.report()
        ref_gdn_rms_report = ref_gdn_rms_many.report()
        ref_attention_rms_report = ref_attention_rms_many.report()
        ref_fast.restore()
        ref_fast = None

        pair.restore_state(engine, initial)
        cand_fast = FastActivationQuantizer(base_quant)
        cand_fast.install()
        cand_many = FastMarshalQuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = cand_many
        cand_repeat = ExactGDNRepeatScale(args.repeat_lib)
        cand_state_batch = ExactGDNStateBatch(args.batch_state_lib)
        cand_gdn_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        cand_attention_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        head_many = ExactRMSNormHeadsMany(args.head_rmsnorm_many_lib)
        (
            cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms,
            cand_residual, cand_attn_gate, cand_repeat_report,
        ) = _run_with_head_rms_many(
            engine, args, cand_repeat, cand_state_batch,
            cand_gdn_rms_many, cand_attention_rms_many, head_many)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)
        cand_many_report = cand_many.report()
        cand_fast_report = cand_fast.report()
        cand_state_batch_report = cand_state_batch.report()
        cand_gdn_rms_report = cand_gdn_rms_many.report()
        cand_attention_rms_report = cand_attention_rms_many.report()
        head_many_report = head_many.report()
        cand_fast.restore()
        cand_fast = None

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            _f32_bytes(a) == _f32_bytes(b) for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("head RMSNorm-many candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"head RMSNorm-many state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        if ref_many_report != EXPECTED_MANY or cand_many_report != EXPECTED_MANY:
            raise RuntimeError(
                f"unexpected quant-many coverage ref={ref_many_report} cand={cand_many_report}")
        if ref_repeat_report != EXPECTED_REPEAT or cand_repeat_report != EXPECTED_REPEAT:
            raise RuntimeError(
                f"unexpected repeat coverage ref={ref_repeat_report} cand={cand_repeat_report}")
        if ref_state_batch_report != EXPECTED_BATCH or cand_state_batch_report != EXPECTED_BATCH:
            raise RuntimeError(
                f"unexpected state-batch coverage ref={ref_state_batch_report} cand={cand_state_batch_report}")
        if int(ref_fast_report["calls"]) != EXPECTED_FAST_QUANT_CALLS:
            raise RuntimeError(f"unexpected reference fast-quantize coverage: {ref_fast_report}")
        if int(cand_fast_report["calls"]) != EXPECTED_FAST_QUANT_CALLS:
            raise RuntimeError(f"unexpected candidate fast-quantize coverage: {cand_fast_report}")
        if ref_gdn_rms_report != EXPECTED_GDN_RMS_MANY or cand_gdn_rms_report != EXPECTED_GDN_RMS_MANY:
            raise RuntimeError(
                f"unexpected GDN RMSNorm-many coverage ref={ref_gdn_rms_report} cand={cand_gdn_rms_report}")
        if ref_attention_rms_report != EXPECTED_ATTN_RMS_MANY or cand_attention_rms_report != EXPECTED_ATTN_RMS_MANY:
            raise RuntimeError(
                f"unexpected attention normal RMSNorm-many coverage ref={ref_attention_rms_report} cand={cand_attention_rms_report}")
        if ref_rms != EXPECTED_REF_WRAPPER:
            raise RuntimeError(f"unexpected reference scalar RMSNorm wrapper coverage: {ref_rms}")
        if cand_rms != EXPECTED_ZERO_WRAPPER:
            raise RuntimeError(f"unexpected candidate scalar RMSNorm wrapper coverage: {cand_rms}")
        if head_many_report != EXPECTED_HEAD_MANY:
            raise RuntimeError(f"unexpected head RMSNorm-many coverage: {head_many_report}")

        ref_total_rows = _combined_rms_rows(
            ref_rms, ref_gdn_rms_report, ref_attention_rms_report)
        ref_total_values = _combined_rms_values(
            ref_rms, ref_gdn_rms_report, ref_attention_rms_report)
        cand_total_rows = _combined_rms_rows(
            cand_rms, cand_gdn_rms_report, cand_attention_rms_report, head_many_report)
        cand_total_values = _combined_rms_values(
            cand_rms, cand_gdn_rms_report, cand_attention_rms_report, head_many_report)
        if (ref_total_rows, cand_total_rows) != (EXPECTED_TOTAL_RMS_ROWS, EXPECTED_TOTAL_RMS_ROWS):
            raise RuntimeError(f"unexpected total RMSNorm rows ref={ref_total_rows} cand={cand_total_rows}")
        if (ref_total_values, cand_total_values) != (EXPECTED_TOTAL_RMS_VALUES, EXPECTED_TOTAL_RMS_VALUES):
            raise RuntimeError(
                f"unexpected total RMSNorm values ref={ref_total_values} cand={cand_total_values}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("head RMSNorm-many A/B requires direct I/O")
        for key, expected in EXPECTED_READER.items():
            if int(reader.get(key, -1)) != expected:
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")

        payload = {
            "schema": "qwen38-attention-head-rmsnorm-many-exact-ab-v1",
            "status": "PASS",
            "claim": "exact sequential many-token Q/K head RMSNorm boundary batching; same-run real GGUF A/B on normal-attention-RMSNorm-many current-best path",
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
            "reference_rmsnorm_remaining": ref_rms,
            "candidate_rmsnorm_remaining": cand_rms,
            "reference_gdn_rmsnorm_many": ref_gdn_rms_report,
            "candidate_gdn_rmsnorm_many": cand_gdn_rms_report,
            "reference_attention_rmsnorm_many": ref_attention_rms_report,
            "candidate_attention_rmsnorm_many": cand_attention_rms_report,
            "attention_head_rmsnorm_many": head_many_report,
            "reference_gdn_state_batch": ref_state_batch_report,
            "candidate_gdn_state_batch": cand_state_batch_report,
            "reference_fast_quantize": ref_fast_report,
            "candidate_fast_quantize": cand_fast_report,
            "reference_quant_many": ref_many_report,
            "candidate_quant_many": cand_many_report,
            "native_f32": native_f32.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "attention_layers": 16,
                "qk_groups_per_layer": 2,
                "rows_per_batch": len(PROMPT_IDS),
                "reference_qk_head_wrapper_calls": EXPECTED_HEAD_MANY["token_rows"],
                "candidate_qk_head_batch_calls": EXPECTED_HEAD_MANY["calls"],
                "normalized_head_rows": EXPECTED_HEAD_MANY["head_rows"],
                "normal_rmsnorm_change": False,
                "head_row_kernel_change": False,
                "head_order_change": False,
                "arithmetic_change": False,
            },
            "baseline_attention_normal_rmsnorm_many_ab_run": 33847120243,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_ATTENTION_HEAD_RMSNORM_MANY_REAL_BITWISE_PASS")
        return payload
    finally:
        if ref_fast is not None:
            ref_fast.restore()
        if cand_fast is not None:
            cand_fast.restore()
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--rmsnorm-lib", type=Path, required=True)
    s.add_argument("--head-rmsnorm-many-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "batch-state-lib",
        "f32-lib", "attn-lib", "conv-lib", "gate-lib", "swiglu-lib",
        "rmsnorm-lib", "rmsnorm-many-lib", "head-rmsnorm-many-lib",
        "residual-lib", "attention-gate-lib", "repeat-lib", "inventory",
        "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.rmsnorm_lib, args.head_rmsnorm_many_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
