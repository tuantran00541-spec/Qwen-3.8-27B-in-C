#!/usr/bin/env python3
"""Bitwise-exact native GDN causal-conv + SiLU A/B gate for Qwen3.8-27B.

Reference and candidate both use the proven staged matvec-many prefill and exact
native causal attention core.  The only candidate change is replacing the
Python 4-tap depthwise GDN convolution + SiLU loop with one native C call per
recurrent layer.  Hidden vectors and all persistent decoder state must match the
known 11-token anchors bit-for-bit.
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
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from attention_core_runtime import ExactAttentionCore
from gdn_conv_silu_runtime import ExactGDNConvSilu
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def _python_conv_reference(qkv_rows, kernels, history):
    hist = [array("f", row) for row in history]
    out = []
    for row in qkv_rows:
        prior = list(hist[-3:])
        conv = [0.0] * len(row)
        for c in range(len(row)):
            cur = gen.mulf(row[c], kernels[c * 4 + 3])
            for lag, old in enumerate(reversed(prior), start=1):
                cur = gen.addf(cur, gen.mulf(old[c], kernels[c * 4 + 3 - lag]))
            conv[c] = gen.t2.siluf(cur)
        out.append(conv)
        hist.append(array("f", row))
        if len(hist) > 3:
            del hist[0]
    return out


def sanity(conv_lib: Path) -> None:
    gen.exact.install()
    core = ExactGDNConvSilu(conv_lib)
    conv_dim = 37
    kernels = [gen.f32((((i * 17 + 5) % 67) - 33) / 29.0) for i in range(conv_dim * 4)]
    for n_hist in (0, 1, 2, 3):
        history = [
            [gen.f32((((r * 31 + c * 7 + 3) % 89) - 44) / 23.0) for c in range(conv_dim)]
            for r in range(n_hist)
        ]
        for n_tokens in (1, 2, 5, 11):
            qkv = [
                [gen.f32((((t * 43 + c * 13 + 11) % 101) - 50) / 19.0) for c in range(conv_dim)]
                for t in range(n_tokens)
            ]
            ref = _python_conv_reference(qkv, kernels, history)
            cand = core.compute(qkv, kernels, history)
            if len(ref) != len(cand) or any(_f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref, cand)):
                raise SystemExit(f"native GDN conv bitwise mismatch hist={n_hist} tokens={n_tokens}")
    print("QWEN38_GDN_CONV_SILU_EXACT_SANITY PASS")


def _recurrent_layer_native_conv(core: ExactGDNConvSilu, stats: dict[str, float], engine, hidden, il, view, metas, vec):
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
    conv_rows = core.compute(qkv, kernels, hist)
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
        core_out = [float(out_buf[i]) for i in range(gdn.VALUE_DIM)]
        core_h = gdn.split_heads(core_out, gdn.V_HEADS)
        z_h = gdn.split_heads(z[j], gdn.V_HEADS)
        gated: list[float] = []
        for ch, zh in zip(core_h, z_h):
            nh = gen.rmswrap.ggml_rms_norm(ch, norm_w, gdn.RMS_EPS)
            gated.extend(gen.mulf(nh[d], gen.t2.siluf(zh[d])) for d in range(gdn.HEAD_DIM))
        gated_rows.append(gated)

    merged = [array("f", row) for row in hist]
    merged.extend(array("f", row) for row in qkv)
    hist[:] = merged[-3:]

    linear = runtime.matvec_many(view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gated_rows)
    residual = [
        [gen.addf(hidden[j][i], linear[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]
    post_norm_w = vec("post_attention_norm.weight")
    post = [gdn.rms_norm(r, post_norm_w) for r in residual]
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    return [
        [gen.addf(residual[j][i], fo[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]


def _run_with_patches(engine, *, attn_core, conv_core=None):
    attn_stats = {"native_attention_core_seconds": 0.0}
    conv_stats = {"native_conv_silu_seconds": 0.0}
    old_attn = prefill._full_attention_layer_many
    old_rec = prefill._recurrent_layer_many

    def native_attn(engine_, hidden, il, pos0, view, metas, vec):
        return attn_probe._full_attention_layer_native(
            engine_, hidden, il, pos0, view, metas, vec, attn_core, attn_stats)

    prefill._full_attention_layer_many = native_attn
    if conv_core is not None:
        def native_rec(engine_, hidden, il, view, metas, vec):
            return _recurrent_layer_native_conv(conv_core, conv_stats, engine_, hidden, il, view, metas, vec)
        prefill._recurrent_layer_many = native_rec
    try:
        before = int(engine.reader.report()["bytes_read"])
        t0 = time.monotonic()
        hidden = prefill.step_block_many(engine, PROMPT_IDS)
        seconds = time.monotonic() - t0
        k3_bytes = int(engine.reader.report()["bytes_read"]) - before
        return hidden, seconds, k3_bytes, attn_stats, conv_stats
    finally:
        prefill._full_attention_layer_many = old_attn
        prefill._recurrent_layer_many = old_rec


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        initial = pair.capture_state(engine)

        ref_attn = ExactAttentionCore(args.attn_lib)
        ref_hidden, ref_seconds, ref_bytes, ref_attn_stats, _ = _run_with_patches(
            engine, attn_core=ref_attn)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        cand_attn = ExactAttentionCore(args.attn_lib)
        conv_core = ExactGDNConvSilu(args.conv_lib)
        cand_hidden, cand_seconds, cand_bytes, cand_attn_stats, conv_stats = _run_with_patches(
            engine, attn_core=cand_attn, conv_core=conv_core)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b) for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("native GDN conv candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native GDN conv candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("native GDN conv A/B requires direct I/O")

        conv_report = conv_core.report()
        if conv_report["calls"] != 48 or conv_report["tokens"] != 48 * len(PROMPT_IDS):
            raise RuntimeError(f"unexpected native conv coverage: {conv_report}")

        payload = {
            "schema": "qwen38-gdn-conv-silu-exact-ab-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
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
            "native_conv_silu_seconds_candidate": conv_stats["native_conv_silu_seconds"],
            "native_conv_silu": conv_report,
            "reference_native_attention_seconds": ref_attn_stats["native_attention_core_seconds"],
            "candidate_native_attention_seconds": cand_attn_stats["native_attention_core_seconds"],
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": engine.reader.report(),
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "research_basis": [
                "Transformers/Qwen uses a dedicated causal_conv1d_update fast path for cached single-token decode.",
                "The exact internal profile identified causal_conv_silu as the largest clearly isolated Python GDN hotspot.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_GDN_CONV_SILU_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--conv-lib", type=Path, required=True)
    r = sub.add_parser("real")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--quant-lib", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--attn-lib", type=Path, required=True)
    r.add_argument("--conv-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.conv_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
