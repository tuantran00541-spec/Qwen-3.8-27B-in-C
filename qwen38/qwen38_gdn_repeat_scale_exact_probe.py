#!/usr/bin/env python3
"""Exact native GDN q/k repeat-scale A/B on the current-best Qwen3.8 path.

Both sides keep the proven Q8 allocation-hoist, native attention sigmoid gate,
native residual adds, RMSNorm, SwiGLU, attention core, and GDN conv/output
helpers. The candidate only replaces Python repeat_k_heads for q/k plus the q
SCALE_GDN multiply with one batched exact C call per recurrent layer.
"""
from __future__ import annotations

import argparse
from array import array
import ctypes
import json
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_attention_gate_exact_probe as attn_gate_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
import qwen38_residual_add_exact_probe as res_probe
from attention_gate_runtime import ExactAttentionGate
from gdn_repeat_scale_runtime import ExactGDNRepeatScale
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many
from residual_add_runtime import ExactResidualAdd

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_RESIDUAL = {"calls": 128, "rows": 1408, "values": 7208960}
EXPECTED_ATTN_GATE = {"calls": 176, "values": 1081344}
EXPECTED_REPEAT = {
    "calls": 48,
    "rows": 48 * len(PROMPT_IDS),
    "q_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "k_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity(repeat_lib: Path) -> None:
    gen.exact.install()
    core = ExactGDNRepeatScale(repeat_lib)
    scale = gen.t2.SCALE_GDN
    for rows, key_dim, repeats in ((1, 1, 3), (3, 17, 3), (4, 128, 3), (11, gdn.KEY_DIM, 3)):
        q_rows = [
            [((j * key_dim + i) * 37 + 11) % 4093 / 31.0 - 66.0 for i in range(key_dim)]
            for j in range(rows)
        ]
        k_rows = [
            [((j * key_dim + i) * 53 + 7) % 3079 / 29.0 - 53.0 for i in range(key_dim)]
            for j in range(rows)
        ]
        ref_q: list[list[float]] = []
        ref_k: list[list[float]] = []
        for q_row, k_row in zip(q_rows, k_rows):
            rq: list[float] = []
            rk: list[float] = []
            for _ in range(repeats):
                rq.extend(gen.mulf(v, scale) for v in q_row)
                rk.extend(k_row)
            ref_q.append(rq)
            ref_k.append(rk)
        cand_q, cand_k = core.compute(q_rows, k_rows, repeats=repeats, scale=scale)
        if any(_f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref_q, cand_q)):
            raise SystemExit(
                f"native GDN q repeat-scale bitwise mismatch rows={rows} key_dim={key_dim}")
        if any(_f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref_k, cand_k)):
            raise SystemExit(
                f"native GDN k repeat bitwise mismatch rows={rows} key_dim={key_dim}")
    print("QWEN38_GDN_REPEAT_SCALE_EXACT_SANITY PASS")


def _recurrent_with_native_repeat(
    conv_core, gate_core, stats: dict[str, float],
    engine, hidden, il, view, metas, vec,
    residual_core: ExactResidualAdd,
    residual_stats: dict[str, float],
    repeat_core: ExactGDNRepeatScale,
    repeat_stats: dict[str, float],
):
    runtime = engine.runtime
    p = f"blk.{il}"
    attn_norm_w = vec("attn_norm.weight")
    xs = [gdn.rms_norm(h, attn_norm_w) for h in hidden]

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

    gated_rows: list[list[float]] = []
    for j in range(len(hidden)):
        beta = [gen.exact.sigmoid_f32(v) for v in beta_raw[j]]
        gate = [
            gen.mulf(aa[h], gen.t2.softplusf(gen.addf(alpha[j][h], dt[h])))
            for h in range(gdn.V_HEADS)
        ]
        v = conv_rows[j][2 * gdn.KEY_DIM :]

        out_buf = (ctypes.c_float * gdn.VALUE_DIM)()
        rc = engine.state_lib.qwen_gdn_ar_step_f32(
            state,
            gen.t2.carr(q48_rows[j]),
            gen.t2.carr(k48_rows[j]),
            gen.t2.carr(v),
            gen.t2.carr(gate),
            gen.t2.carr(beta),
            out_buf,
        )
        if rc != 0:
            raise RuntimeError(f"layer {il}: GDN state kernel rc={rc}")

        t0 = time.monotonic()
        gated = gate_core.compute(
            out_buf,
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
    post = [gdn.rms_norm(r, post_norm_w) for r in residual]
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    t0 = time.monotonic()
    out = residual_core.compute(residual, fo)
    residual_stats["native_residual_add_seconds"] += time.monotonic() - t0
    return out


def _current_best_run(
    engine,
    args,
    *,
    repeat_core: ExactGDNRepeatScale | None,
):
    residual_core = ExactResidualAdd(args.residual_lib)
    attention_gate_core = ExactAttentionGate(args.attention_gate_lib)
    attention_gate_stats = {"native_attention_gate_seconds": 0.0}
    repeat_stats = {"native_gdn_repeat_scale_seconds": 0.0}
    old_attn = res_probe._full_attention_with_residual
    old_rec = res_probe._recurrent_with_residual

    def attention_impl(
        engine_, hidden, il, pos0, view, metas, vec, core, stats,
        residual_core_, residual_stats,
    ):
        return attn_gate_probe._full_attention_with_native_gate(
            engine_, hidden, il, pos0, view, metas, vec, core, stats,
            residual_core_, residual_stats,
            attention_gate_core, attention_gate_stats,
        )

    res_probe._full_attention_with_residual = attention_impl
    if repeat_core is not None:
        def recurrent_impl(
            conv_core, gate_core, stats, engine_, hidden, il, view, metas, vec,
            residual_core_, residual_stats,
        ):
            return _recurrent_with_native_repeat(
                conv_core, gate_core, stats, engine_, hidden, il, view, metas, vec,
                residual_core_, residual_stats, repeat_core, repeat_stats,
            )
        res_probe._recurrent_with_residual = recurrent_impl

    try:
        hidden, seconds, k3_bytes, stats, rms = res_probe._best_run(
            engine, args, residual_core=residual_core)
    finally:
        res_probe._full_attention_with_residual = old_attn
        res_probe._recurrent_with_residual = old_rec

    stats = dict(stats)
    stats.update(attention_gate_stats)
    stats.update(repeat_stats)
    return (
        hidden,
        seconds,
        k3_bytes,
        stats,
        rms,
        residual_core.report(),
        attention_gate_core.report(),
        None if repeat_core is None else repeat_core.report(),
    )


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        initial = pair.capture_state(engine)

        (
            ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms,
            ref_residual, ref_attn_gate, _,
        ) = _current_best_run(engine, args, repeat_core=None)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        repeat_core = ExactGDNRepeatScale(args.repeat_lib)
        (
            cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms,
            cand_residual, cand_attn_gate, cand_repeat,
        ) = _current_best_run(engine, args, repeat_core=repeat_core)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("native GDN repeat-scale candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native GDN repeat-scale state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("native GDN repeat-scale A/B requires direct I/O")
        if ref_residual != EXPECTED_RESIDUAL or cand_residual != EXPECTED_RESIDUAL:
            raise RuntimeError(f"unexpected residual coverage ref={ref_residual} cand={cand_residual}")
        if ref_attn_gate != EXPECTED_ATTN_GATE or cand_attn_gate != EXPECTED_ATTN_GATE:
            raise RuntimeError(
                f"unexpected attention-gate coverage ref={ref_attn_gate} cand={cand_attn_gate}")
        if ref_rms.get("total_rows") != 6336 or cand_rms.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected RMSNorm coverage ref={ref_rms} cand={cand_rms}")
        if cand_repeat != EXPECTED_REPEAT:
            raise RuntimeError(f"unexpected GDN repeat-scale coverage {cand_repeat}")

        payload = {
            "schema": "qwen38-gdn-repeat-scale-exact-ab-v1",
            "status": "PASS",
            "claim": "exact native GDN q/k repeat plus q scale; same-run real GGUF A/B on attention-gate current-best path",
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
            "reference_residual": ref_residual,
            "candidate_residual": cand_residual,
            "reference_attention_gate": ref_attn_gate,
            "candidate_attention_gate": cand_attn_gate,
            "reference_rmsnorm": ref_rms,
            "candidate_rmsnorm": cand_rms,
            "gdn_repeat_scale": cand_repeat,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "recurrent_layers": 48,
                "rows_per_prefill": 48 * len(PROMPT_IDS),
                "q_scaled_repeat_values": EXPECTED_REPEAT["q_values"],
                "k_repeat_values": EXPECTED_REPEAT["k_values"],
                "reference": "Python repeat_k_heads for q/k plus mulf SCALE_GDN for q",
                "candidate": "one batched exact native repeat-scale call per recurrent layer",
                "arithmetic_change": False,
            },
            "baseline_attention_gate_ab_run": 33839504070,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_GDN_REPEAT_SCALE_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--repeat-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib", "residual-lib",
        "attention-gate-lib", "repeat-lib", "inventory", "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.repeat_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
