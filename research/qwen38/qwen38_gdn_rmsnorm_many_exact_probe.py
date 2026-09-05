#!/usr/bin/env python3
"""Exact GDN many-row RMSNorm A/B on current-best Qwen3.8.

Reference and candidate both include fast activation quantize, fast matvec output
marshal, Q8 noalloc, native GDN repeat-scale, and exact sequential GDN state
batching.  Candidate only batches the two normal width-5120 RMSNorm sites in
each of the 48 recurrent layers.  Attention-layer normal/head RMSNorm is left
unchanged in this experiment.
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
import qwen38_gdn_state_batch_exact_probe as batch_probe
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
EXPECTED_MANY = {"many_calls": 400, "many_vectors": 4400, "many_rows": 42893312}
EXPECTED_REPEAT = {
    "calls": 48,
    "rows": 48 * len(PROMPT_IDS),
    "q_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "k_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
}
EXPECTED_BATCH = {
    "calls": 48,
    "rows": 48 * len(PROMPT_IDS),
    "q_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "k_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "v_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "gate_values": 48 * len(PROMPT_IDS) * gdn.V_HEADS,
    "beta_values": 48 * len(PROMPT_IDS) * gdn.V_HEADS,
    "output_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
}
EXPECTED_FAST_QUANT_CALLS = 3696
EXPECTED_RMS_MANY = {
    "calls": 48 * 2,
    "rows": 48 * len(PROMPT_IDS) * 2,
    "values": 48 * len(PROMPT_IDS) * 2 * gdn.HIDDEN,
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity(rmsnorm_lib: Path, rmsnorm_many_lib: Path) -> None:
    scalar = ExactRMSNorm(rmsnorm_lib)
    many = ExactRMSNormMany(rmsnorm_many_lib)
    for rows, width in ((3, 17), (4, 128), (3, gdn.HIDDEN)):
        values = [
            [(((r * width + i) * 104729 + 1543) % 200003 - 100001) / 4096.0
             for i in range(width)]
            for r in range(rows)
        ]
        weight = [(((i * 8191 + 37) % 32749) - 16374) / 8192.0 for i in range(width)]
        ref = [scalar.compute(row, weight, gdn.RMS_EPS) for row in values]
        cand = many.compute(values, weight, gdn.RMS_EPS)
        if len(ref) != len(cand) or any(
            _f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref, cand)
        ):
            raise SystemExit(f"RMSNorm-many bitwise mismatch rows={rows} width={width}")
    print("QWEN38_GDN_RMSNORM_MANY_EXACT_SANITY PASS")


def _recurrent_with_rms_many(
    conv_core, gate_core, stats: dict[str, float],
    engine, hidden, il, view, metas, vec,
    residual_core, residual_stats,
    repeat_core: ExactGDNRepeatScale,
    repeat_stats: dict[str, float],
    state_batch_core: ExactGDNStateBatch,
    state_batch_stats: dict[str, float],
    rms_many: ExactRMSNormMany,
    rms_many_stats: dict[str, float],
):
    runtime = engine.runtime
    p = f"blk.{il}"
    attn_norm_w = vec("attn_norm.weight")
    t0 = time.monotonic()
    xs = rms_many.compute(hidden, attn_norm_w, gdn.RMS_EPS)
    rms_many_stats["native_gdn_rmsnorm_many_seconds"] += time.monotonic() - t0

    qkv = runtime.matvec_many(view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], xs)
    z = runtime.matvec_many(view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], xs)
    beta_raw = runtime.matvec_many(view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], xs)
    alpha = runtime.matvec_many(view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], xs)

    dt = vec("ssm_dt.bias")
    aa = vec("ssm_a")
    kernels = vec("ssm_conv1d.weight")
    norm_w = vec("ssm_norm.weight")
    state = engine.states[il]
    hist = engine.conv_history[il]

    t0 = time.monotonic()
    conv_rows = conv_core.compute(qkv, kernels, hist)
    stats["native_conv_silu_seconds"] += time.monotonic() - t0

    qn_rows: list[list[float]] = []
    kn_rows: list[list[float]] = []
    for conv in conv_rows:
        q = conv[: gdn.KEY_DIM]
        k = conv[gdn.KEY_DIM : 2 * gdn.KEY_DIM]
        qn_rows.append(gdn.flatten([
            gdn.l2_norm(h) for h in gdn.split_heads(q, gdn.K_HEADS)
        ]))
        kn_rows.append(gdn.flatten([
            gdn.l2_norm(h) for h in gdn.split_heads(k, gdn.K_HEADS)
        ]))

    t0 = time.monotonic()
    q48_rows, k48_rows = repeat_core.compute(
        qn_rows,
        kn_rows,
        repeats=gdn.V_HEADS // gdn.K_HEADS,
        scale=gen.t2.SCALE_GDN,
    )
    repeat_stats["native_gdn_repeat_scale_seconds"] += time.monotonic() - t0

    v_rows: list[list[float]] = []
    gate_rows: list[list[float]] = []
    beta_rows: list[list[float]] = []
    for j in range(len(hidden)):
        beta_rows.append([gen.exact.sigmoid_f32(v) for v in beta_raw[j]])
        gate_rows.append([
            gen.mulf(aa[h], gen.t2.softplusf(gen.addf(alpha[j][h], dt[h])))
            for h in range(gdn.V_HEADS)
        ])
        v_rows.append(conv_rows[j][2 * gdn.KEY_DIM :])

    t0 = time.monotonic()
    core_rows = state_batch_core.compute(
        state, q48_rows, k48_rows, v_rows, gate_rows, beta_rows)
    state_batch_stats["native_gdn_state_batch_seconds"] += time.monotonic() - t0

    gated_rows: list[list[float]] = []
    for j in range(len(hidden)):
        t0 = time.monotonic()
        gated = gate_core.compute(
            core_rows[j],
            z[j],
            norm_w,
            heads=gdn.V_HEADS,
            head_dim=gdn.HEAD_DIM,
            eps=gdn.RMS_EPS,
        )
        stats["native_output_gate_seconds"] += time.monotonic() - t0
        gated_rows.append(gated)

    merged = [array("f", row) for row in hist]
    merged.extend(array("f", row) for row in qkv)
    hist[:] = merged[-3:]

    linear = runtime.matvec_many(
        view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gated_rows)
    t0 = time.monotonic()
    residual = residual_core.compute(hidden, linear)
    residual_stats["native_residual_add_seconds"] += time.monotonic() - t0
    post_norm_w = vec("post_attention_norm.weight")
    t0 = time.monotonic()
    post = rms_many.compute(residual, post_norm_w, gdn.RMS_EPS)
    rms_many_stats["native_gdn_rmsnorm_many_seconds"] += time.monotonic() - t0
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    t0 = time.monotonic()
    out = residual_core.compute(residual, fo)
    residual_stats["native_residual_add_seconds"] += time.monotonic() - t0
    return out


def _run_candidate_with_rms_many(
    engine,
    args,
    repeat_core: ExactGDNRepeatScale,
    state_batch_core: ExactGDNStateBatch,
    rms_many: ExactRMSNormMany,
):
    old = batch_probe._recurrent_with_batch_state
    rms_many_stats = {"native_gdn_rmsnorm_many_seconds": 0.0}

    def impl(
        conv_core, gate_core, stats, engine_, hidden, il, view, metas, vec,
        residual_core, residual_stats, repeat_core_, repeat_stats,
        state_batch_core_, state_batch_stats,
    ):
        return _recurrent_with_rms_many(
            conv_core, gate_core, stats,
            engine_, hidden, il, view, metas, vec,
            residual_core, residual_stats,
            repeat_core_, repeat_stats,
            state_batch_core_, state_batch_stats,
            rms_many, rms_many_stats,
        )

    batch_probe._recurrent_with_batch_state = impl
    try:
        result = batch_probe._run_candidate(engine, args, repeat_core, state_batch_core)
    finally:
        batch_probe._recurrent_with_batch_state = old
    hidden, seconds, k3_bytes, stats, rms, residual, attn_gate, repeat_report = result
    stats = dict(stats)
    stats.update(rms_many_stats)
    return hidden, seconds, k3_bytes, stats, rms, residual, attn_gate, repeat_report


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
        (
            ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms,
            ref_residual, ref_attn_gate, ref_repeat_report,
        ) = batch_probe._run_candidate(engine, args, ref_repeat, ref_state_batch)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)
        ref_many_report = ref_many.report()
        ref_fast_report = ref_fast.report()
        ref_state_batch_report = ref_state_batch.report()
        ref_fast.restore()
        ref_fast = None

        pair.restore_state(engine, initial)
        cand_fast = FastActivationQuantizer(base_quant)
        cand_fast.install()
        cand_many = FastMarshalQuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = cand_many
        cand_repeat = ExactGDNRepeatScale(args.repeat_lib)
        cand_state_batch = ExactGDNStateBatch(args.batch_state_lib)
        rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        (
            cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms,
            cand_residual, cand_attn_gate, cand_repeat_report,
        ) = _run_candidate_with_rms_many(
            engine, args, cand_repeat, cand_state_batch, rms_many)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)
        cand_many_report = cand_many.report()
        cand_fast_report = cand_fast.report()
        cand_state_batch_report = cand_state_batch.report()
        rms_many_report = rms_many.report()
        cand_fast.restore()
        cand_fast = None

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            _f32_bytes(a) == _f32_bytes(b) for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("GDN RMSNorm-many candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"GDN RMSNorm-many state mismatch: {state_mismatch}")
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
        if rms_many_report != EXPECTED_RMS_MANY:
            raise RuntimeError(f"unexpected GDN RMSNorm-many coverage: {rms_many_report}")
        if ref_rms.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected reference RMSNorm coverage: {ref_rms}")
        if int(cand_rms.get("total_rows", -1)) + rms_many_report["rows"] != 6336:
            raise RuntimeError(
                f"unexpected candidate combined RMSNorm coverage native={cand_rms} many={rms_many_report}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("GDN RMSNorm-many A/B requires direct I/O")

        payload = {
            "schema": "qwen38-gdn-rmsnorm-many-exact-ab-v1",
            "status": "PASS",
            "claim": "exact sequential many-row RMSNorm in recurrent layers; same-run real GGUF A/B on batch-state current-best path",
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
            "reference_rmsnorm": ref_rms,
            "candidate_rmsnorm_remaining": cand_rms,
            "gdn_rmsnorm_many": rms_many_report,
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
                "reference_recurrent_normal_rms_calls": 48 * len(PROMPT_IDS) * 2,
                "candidate_recurrent_normal_rms_calls": 48 * 2,
                "rows_per_batch": len(PROMPT_IDS),
                "attention_rmsnorm_change": False,
                "rmsnorm_row_kernel_change": False,
                "arithmetic_change": False,
            },
            "baseline_gdn_state_batch_ab_run": 33844318721,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_GDN_RMSNORM_MANY_REAL_BITWISE_PASS")
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
