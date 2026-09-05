#!/usr/bin/env python3
"""Exact full-attention normal RMSNorm batching A/B on current-best Qwen3.8.

Both sides include the current GDN many-row RMSNorm path plus fast activation
quantize, fast matvec output marshal, Q8 noalloc, native residual add,
attention sigmoid gate, GDN repeat-scale, sequential GDN state batching, and
all previously proven native helpers. Candidate changes only the two normal
width-5120 RMSNorm sites in each of the 16 full-attention layers. Q/K head
RMSNorm remains on the existing exact per-token wrapper in this experiment.
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
import qwen38_attention_gate_exact_probe as attn_gate_probe
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
EXPECTED_ATTN_RMS_MANY = {
    "calls": 16 * 2,
    "rows": 16 * len(PROMPT_IDS) * 2,
    "values": 16 * len(PROMPT_IDS) * 2 * gdn.HIDDEN,
}
EXPECTED_REF_RMS = {
    "calls": 16 * len(PROMPT_IDS) * 2,
    "rows": 16 * len(PROMPT_IDS) * 2,
    "values": 16 * len(PROMPT_IDS) * 2 * gdn.HIDDEN,
    "head_calls": 16 * len(PROMPT_IDS) * 2,
    "head_rows": 16 * len(PROMPT_IDS) * (gen.attn.N_HEAD + gen.attn.N_HEAD_KV),
    "head_values": 16 * len(PROMPT_IDS) * (gen.attn.Q_DIM + gen.attn.KV_DIM),
}
EXPECTED_TOTAL_RMS_ROWS = 6336
EXPECTED_TOTAL_RMS_VALUES = 8470528
EXPECTED_READER = {
    "ring_slots": 2,
    "slot_bytes": 336449536,
    "planned_bytes": 672899072,
    "budget_bytes": 672899072,
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity(rmsnorm_lib: Path, rmsnorm_many_lib: Path) -> None:
    scalar = ExactRMSNorm(rmsnorm_lib)
    many = ExactRMSNormMany(rmsnorm_many_lib)
    rows = len(PROMPT_IDS)
    width = gdn.HIDDEN
    for site, salt in (("attn_norm", 17), ("post_attention_norm", 31)):
        values = [
            [
                (((r * width + i) * 104729 + salt * 1543) % 200003 - 100001)
                / 4096.0
                for i in range(width)
            ]
            for r in range(rows)
        ]
        weight = [
            (((i * 8191 + salt * 37) % 32749) - 16374) / 8192.0
            for i in range(width)
        ]
        ref = [scalar.compute(row, weight, gdn.RMS_EPS) for row in values]
        cand = many.compute(values, weight, gdn.RMS_EPS)
        if len(ref) != len(cand) or any(
            _f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref, cand)
        ):
            raise SystemExit(f"attention RMSNorm-many bitwise mismatch site={site}")
    expected = {"calls": 2, "rows": rows * 2, "values": rows * width * 2}
    if many.report() != expected:
        raise SystemExit(f"unexpected attention RMSNorm-many sanity coverage: {many.report()}")
    print("QWEN38_ATTENTION_RMSNORM_MANY_EXACT_SANITY PASS")


def _full_attention_with_normal_rms_many(
    engine, hidden, il: int, pos0: int, view, metas, vec,
    core, stats: dict[str, float],
    residual_core, residual_stats: dict[str, float],
    attention_gate_core, attention_gate_stats: dict[str, float],
    rms_many: ExactRMSNormMany,
    rms_many_stats: dict[str, float],
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
    t0 = time.monotonic()
    post = rms_many.compute(residual, post_norm_w, gdn.RMS_EPS)
    rms_many_stats["native_attention_rmsnorm_many_seconds"] += time.monotonic() - t0
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    t0 = time.monotonic()
    out = residual_core.compute(residual, fo)
    residual_stats["native_residual_add_seconds"] += time.monotonic() - t0
    return out


def _run_with_attention_rms_many(
    engine,
    args,
    repeat_core: ExactGDNRepeatScale,
    state_batch_core: ExactGDNStateBatch,
    gdn_rms_many: ExactRMSNormMany,
    attention_rms_many: ExactRMSNormMany,
):
    old = attn_gate_probe._full_attention_with_native_gate
    attention_rms_stats = {"native_attention_rmsnorm_many_seconds": 0.0}

    def impl(
        engine_, hidden, il, pos0, view, metas, vec, core, stats,
        residual_core, residual_stats, attention_gate_core, attention_gate_stats,
    ):
        return _full_attention_with_normal_rms_many(
            engine_, hidden, il, pos0, view, metas, vec, core, stats,
            residual_core, residual_stats, attention_gate_core, attention_gate_stats,
            attention_rms_many, attention_rms_stats,
        )

    attn_gate_probe._full_attention_with_native_gate = impl
    try:
        result = gdn_rms_probe._run_candidate_with_rms_many(
            engine, args, repeat_core, state_batch_core, gdn_rms_many)
    finally:
        attn_gate_probe._full_attention_with_native_gate = old
    hidden, seconds, k3_bytes, stats, rms, residual, attn_gate, repeat_report = result
    stats = dict(stats)
    stats.update(attention_rms_stats)
    return hidden, seconds, k3_bytes, stats, rms, residual, attn_gate, repeat_report


def _check_rms_wrapper(report: dict[str, int], *, normal_rows: int) -> None:
    expected_normal_values = normal_rows * gdn.HIDDEN
    if report.get("calls") != normal_rows or report.get("rows") != normal_rows:
        raise RuntimeError(f"unexpected normal RMSNorm wrapper coverage: {report}")
    if report.get("values") != expected_normal_values:
        raise RuntimeError(f"unexpected normal RMSNorm wrapper values: {report}")
    for key in ("head_calls", "head_rows", "head_values"):
        if report.get(key) != EXPECTED_REF_RMS[key]:
            raise RuntimeError(f"head RMSNorm changed at {key}: {report}")


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
        (
            ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms,
            ref_residual, ref_attn_gate, ref_repeat_report,
        ) = gdn_rms_probe._run_candidate_with_rms_many(
            engine, args, ref_repeat, ref_state_batch, ref_gdn_rms_many)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)
        ref_many_report = ref_many.report()
        ref_fast_report = ref_fast.report()
        ref_state_batch_report = ref_state_batch.report()
        ref_gdn_rms_report = ref_gdn_rms_many.report()
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
        attention_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        (
            cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms,
            cand_residual, cand_attn_gate, cand_repeat_report,
        ) = _run_with_attention_rms_many(
            engine, args, cand_repeat, cand_state_batch,
            cand_gdn_rms_many, attention_rms_many)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)
        cand_many_report = cand_many.report()
        cand_fast_report = cand_fast.report()
        cand_state_batch_report = cand_state_batch.report()
        cand_gdn_rms_report = cand_gdn_rms_many.report()
        attention_rms_report = attention_rms_many.report()
        cand_fast.restore()
        cand_fast = None

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            _f32_bytes(a) == _f32_bytes(b) for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("attention RMSNorm-many candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"attention RMSNorm-many state mismatch: {state_mismatch}")
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
        if ref_gdn_rms_report != EXPECTED_GDN_RMS_MANY:
            raise RuntimeError(f"unexpected reference GDN RMSNorm-many coverage: {ref_gdn_rms_report}")
        if cand_gdn_rms_report != EXPECTED_GDN_RMS_MANY:
            raise RuntimeError(f"unexpected candidate GDN RMSNorm-many coverage: {cand_gdn_rms_report}")
        if attention_rms_report != EXPECTED_ATTN_RMS_MANY:
            raise RuntimeError(f"unexpected attention RMSNorm-many coverage: {attention_rms_report}")

        _check_rms_wrapper(ref_rms, normal_rows=16 * len(PROMPT_IDS) * 2)
        _check_rms_wrapper(cand_rms, normal_rows=0)
        ref_total_rows = int(ref_rms["total_rows"]) + ref_gdn_rms_report["rows"]
        ref_total_values = int(ref_rms["total_values"]) + ref_gdn_rms_report["values"]
        cand_total_rows = (
            int(cand_rms["total_rows"])
            + cand_gdn_rms_report["rows"]
            + attention_rms_report["rows"]
        )
        cand_total_values = (
            int(cand_rms["total_values"])
            + cand_gdn_rms_report["values"]
            + attention_rms_report["values"]
        )
        if (ref_total_rows, cand_total_rows) != (EXPECTED_TOTAL_RMS_ROWS, EXPECTED_TOTAL_RMS_ROWS):
            raise RuntimeError(f"unexpected total RMSNorm rows ref={ref_total_rows} cand={cand_total_rows}")
        if (ref_total_values, cand_total_values) != (EXPECTED_TOTAL_RMS_VALUES, EXPECTED_TOTAL_RMS_VALUES):
            raise RuntimeError(
                f"unexpected total RMSNorm values ref={ref_total_values} cand={cand_total_values}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("attention RMSNorm-many A/B requires direct I/O")
        for key, expected in EXPECTED_READER.items():
            if int(reader.get(key, -1)) != expected:
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")

        payload = {
            "schema": "qwen38-attention-rmsnorm-many-exact-ab-v1",
            "status": "PASS",
            "claim": "exact sequential many-row RMSNorm for normal full-attention sites; same-run real GGUF A/B on current GDN RMSNorm-many stack",
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
            "attention_rmsnorm_many": attention_rms_report,
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
                "normal_rms_sites_per_layer": 2,
                "rows_per_batch": len(PROMPT_IDS),
                "reference_attention_normal_rms_calls": EXPECTED_ATTN_RMS_MANY["rows"],
                "candidate_attention_normal_rms_calls": EXPECTED_ATTN_RMS_MANY["calls"],
                "head_rmsnorm_change": False,
                "gdn_rmsnorm_change": False,
                "rmsnorm_row_kernel_change": False,
                "arithmetic_change": False,
            },
            "baseline_gdn_rmsnorm_many_ab_run": 33845199757,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_ATTENTION_RMSNORM_MANY_REAL_BITWISE_PASS")
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
    s.add_argument("--rmsnorm-many-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "batch-state-lib",
        "f32-lib", "attn-lib", "conv-lib", "gate-lib", "swiglu-lib",
        "rmsnorm-lib", "rmsnorm-many-lib", "residual-lib",
        "attention-gate-lib", "repeat-lib", "inventory", "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.rmsnorm_lib, args.rmsnorm_many_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
