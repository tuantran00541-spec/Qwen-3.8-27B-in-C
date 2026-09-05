#!/usr/bin/env python3
"""Same-run real A/B for exact native Qwen3.8 residual additions.

Both sides use the current-best exact staged-prefill stack including native
RMSNorm and SwiGLU. The candidate only replaces the two HIDDEN-wide residual
adds in every decoder layer with one batched C call per add site.
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
import qwen38_attention_core_exact_probe as attn_probe
import qwen38_gdn_output_gate_exact_probe as gate_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
import qwen38_swiglu_exact_probe as sw_probe
from attention_core_runtime import ExactAttentionCore
from gdn_conv_silu_runtime import ExactGDNConvSilu
from gdn_output_gate_runtime import ExactGDNOutputGate
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many
from residual_add_runtime import ExactResidualAdd
from rmsnorm_runtime import ExactRMSNorm
from swiglu_runtime import ExactSwiGLU

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_CALLS = 64 * 2
EXPECTED_ROWS = EXPECTED_CALLS * len(PROMPT_IDS)
EXPECTED_VALUES = EXPECTED_ROWS * gdn.HIDDEN


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity(residual_lib: Path) -> None:
    gen.exact.install()
    core = ExactResidualAdd(residual_lib)
    for rows, width in ((1, 1), (2, 3), (3, 17), (11, 128), (11, gdn.HIDDEN)):
        a = [[gen.f32((((j * width + i) * 37 + 11) % 4093 - 2046) / 31.0)
              for i in range(width)] for j in range(rows)]
        b = [[gen.f32((((j * width + i) * 53 + 7) % 3079 - 1539) / 29.0)
              for i in range(width)] for j in range(rows)]
        ref = [[gen.addf(a[j][i], b[j][i]) for i in range(width)] for j in range(rows)]
        cand = core.compute(a, b)
        if len(ref) != len(cand) or any(
            _f32_bytes(x) != _f32_bytes(y) for x, y in zip(ref, cand)
        ):
            raise SystemExit(f"native residual-add bitwise mismatch rows={rows} width={width}")
    print("QWEN38_RESIDUAL_ADD_EXACT_SANITY PASS")


def _full_attention_with_residual(
    engine, hidden, il: int, pos0: int, view, metas, vec,
    core: ExactAttentionCore, stats: dict[str, float],
    residual_core: ExactResidualAdd, residual_stats: dict[str, float],
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
        gs = [gen.exact.sigmoid_f32(vv) for vv in gate]
        gated_rows.append([
            gen.mulf(pregate[i], gs[i]) for i in range(gen.attn.Q_DIM)
        ])

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


def _recurrent_with_residual(
    conv_core: ExactGDNConvSilu,
    gate_core: ExactGDNOutputGate,
    stats: dict[str, float],
    engine, hidden, il, view, metas, vec,
    residual_core: ExactResidualAdd,
    residual_stats: dict[str, float],
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

    gated_rows: list[list[float]] = []
    for j in range(len(hidden)):
        beta = [gen.exact.sigmoid_f32(v) for v in beta_raw[j]]
        gate = [
            gen.mulf(aa[h], gen.t2.softplusf(gen.addf(alpha[j][h], dt[h])))
            for h in range(gdn.V_HEADS)
        ]

        conv = conv_rows[j]
        q = conv[: gdn.KEY_DIM]
        k = conv[gdn.KEY_DIM : 2 * gdn.KEY_DIM]
        v = conv[2 * gdn.KEY_DIM :]
        qn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(q, gdn.K_HEADS)])
        kn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(k, gdn.K_HEADS)])
        q48 = [gen.mulf(vv, gen.t2.SCALE_GDN) for vv in gen.t2.repeat_k_heads(qn)]
        k48 = gen.t2.repeat_k_heads(kn)

        out_buf = (ctypes.c_float * gdn.VALUE_DIM)()
        rc = engine.state_lib.qwen_gdn_ar_step_f32(
            state,
            gen.t2.carr(q48),
            gen.t2.carr(k48),
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


def _best_run(engine, args, *, residual_core: ExactResidualAdd | None):
    rms_core = ExactRMSNorm(args.rmsnorm_lib)
    old_rms = gdn.rms_norm
    old_heads = gen.attn.rms_norm_heads
    old_attn_impl = attn_probe._full_attention_layer_native
    old_rec_impl = gate_probe._recurrent_layer_native_conv_gate
    rms_seconds = 0.0
    residual_stats = {"native_residual_add_seconds": 0.0}

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

    if residual_core is not None:
        def cand_attn(engine_, hidden, il, pos0, view, metas, vec, core, stats):
            return _full_attention_with_residual(
                engine_, hidden, il, pos0, view, metas, vec, core, stats,
                residual_core, residual_stats)

        def cand_rec(conv_core, gate_core, stats, engine_, hidden, il, view, metas, vec):
            return _recurrent_with_residual(
                conv_core, gate_core, stats, engine_, hidden, il, view, metas, vec,
                residual_core, residual_stats)

        attn_probe._full_attention_layer_native = cand_attn
        gate_probe._recurrent_layer_native_conv_gate = cand_rec

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
        stats.update(residual_stats)
        return hidden, seconds, k3_bytes, stats, rms_core.report()
    finally:
        attn_probe._full_attention_layer_native = old_attn_impl
        gate_probe._recurrent_layer_native_conv_gate = old_rec_impl
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

        ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms = _best_run(
            engine, args, residual_core=None)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        residual_core = ExactResidualAdd(args.residual_lib)
        cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms = _best_run(
            engine, args, residual_core=residual_core)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("native residual-add candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native residual-add candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("native residual-add A/B requires direct I/O")

        report = residual_core.report()
        expected = {
            "calls": EXPECTED_CALLS,
            "rows": EXPECTED_ROWS,
            "values": EXPECTED_VALUES,
        }
        if report != expected:
            raise RuntimeError(f"unexpected residual-add coverage: {report}")
        if ref_rms.get("total_rows") != 6336 or cand_rms.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected RMSNorm coverage ref={ref_rms} cand={cand_rms}")

        payload = {
            "schema": "qwen38-residual-add-exact-ab-v1",
            "status": "PASS",
            "claim": "exact native batched residual additions; same-run real GGUF A/B after native RMSNorm",
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
            "native_residual_add": report,
            "reference_rmsnorm": ref_rms,
            "candidate_rmsnorm": cand_rms,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "profile_basis": {
                "run": 33827573722,
                "artifact": 9920744552,
                "clean_prefill_seconds": 56.092989,
                "python_addf_calls_under_cprofile": 14755328,
                "candidate_residual_values": EXPECTED_VALUES,
                "note": "cProfile ranks Python scalar residual work; same-run A/B here is the speed evidence.",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_RESIDUAL_ADD_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--residual-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib", "residual-lib",
        "inventory", "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.residual_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
