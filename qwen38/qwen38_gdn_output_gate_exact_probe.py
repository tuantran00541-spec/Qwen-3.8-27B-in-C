#!/usr/bin/env python3
"""Real bitwise A/B gate for exact native GDN output RMSNorm + SiLU gating.

Both sides use the already-proven native causal attention and native GDN
causal-conv + SiLU paths.  The only candidate change is fusing the recurrent
state output unpack, per-value-head exact RMSNorm, and z SiLU gating into one C
call per token/layer.
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
import qwen35_k3_full64_ggml_rmsnorm as rmswrap
import qwen38_attention_core_exact_probe as attn_probe
import qwen38_gdn_conv_silu_exact_probe as conv_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from attention_core_runtime import ExactAttentionCore
from gdn_conv_silu_runtime import ExactGDNConvSilu
from gdn_output_gate_runtime import ExactGDNOutputGate
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


def _python_gate_reference(core, z, weight, heads: int, head_dim: int, eps: float):
    out: list[float] = []
    for h in range(heads):
        base = h * head_dim
        ch = [float(core[base + d]) for d in range(head_dim)]
        zh = [float(z[base + d]) for d in range(head_dim)]
        nh = rmswrap.ggml_rms_norm(ch, weight, eps)
        out.extend(gen.mulf(nh[d], gen.t2.siluf(zh[d])) for d in range(head_dim))
    return out


def sanity(gate_lib: Path) -> None:
    gen.exact.install()
    core_impl = ExactGDNOutputGate(gate_lib)
    for head_dim in (1, 3, 17, 128):
        heads = 3
        n = heads * head_dim
        core = [gen.f32((((i * 37 + head_dim * 11) % 257) - 128) / 31.0) for i in range(n)]
        z = [gen.f32((((i * 53 + 7) % 401) - 200) / 10.0) for i in range(n)]
        weight = [gen.f32((((i * 19 + 5) % 113) - 56) / 29.0) for i in range(head_dim)]
        ref = _python_gate_reference(core, z, weight, heads, head_dim, gdn.RMS_EPS)
        core_ct = (ctypes.c_float * n)(*map(float, core))
        cand = core_impl.compute(
            core_ct,
            z,
            weight,
            heads=heads,
            head_dim=head_dim,
            eps=gdn.RMS_EPS,
        )
        if _f32_bytes(ref) != _f32_bytes(cand):
            raise SystemExit(f"native GDN output gate bitwise mismatch head_dim={head_dim}")
    print("QWEN38_GDN_OUTPUT_GATE_EXACT_SANITY PASS")


def _recurrent_layer_native_conv_gate(
    conv_core: ExactGDNConvSilu,
    gate_core: ExactGDNOutputGate,
    stats: dict[str, float],
    engine,
    hidden,
    il,
    view,
    metas,
    vec,
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


def _run_with_patches(engine, *, attn_core, conv_core, gate_core=None):
    attn_stats = {"native_attention_core_seconds": 0.0}
    conv_stats = {"native_conv_silu_seconds": 0.0}
    gate_stats = {"native_output_gate_seconds": 0.0}
    old_attn = prefill._full_attention_layer_many
    old_rec = prefill._recurrent_layer_many

    def native_attn(engine_, hidden, il, pos0, view, metas, vec):
        return attn_probe._full_attention_layer_native(
            engine_, hidden, il, pos0, view, metas, vec, attn_core, attn_stats)

    if gate_core is None:
        def native_rec(engine_, hidden, il, view, metas, vec):
            return conv_probe._recurrent_layer_native_conv(
                conv_core, conv_stats, engine_, hidden, il, view, metas, vec)
    else:
        def native_rec(engine_, hidden, il, view, metas, vec):
            merged_stats = {
                "native_conv_silu_seconds": conv_stats["native_conv_silu_seconds"],
                "native_output_gate_seconds": gate_stats["native_output_gate_seconds"],
            }
            out = _recurrent_layer_native_conv_gate(
                conv_core, gate_core, merged_stats, engine_, hidden, il, view, metas, vec)
            conv_stats["native_conv_silu_seconds"] = merged_stats["native_conv_silu_seconds"]
            gate_stats["native_output_gate_seconds"] = merged_stats["native_output_gate_seconds"]
            return out

    prefill._full_attention_layer_many = native_attn
    prefill._recurrent_layer_many = native_rec
    try:
        before = int(engine.reader.report()["bytes_read"])
        t0 = time.monotonic()
        hidden = prefill.step_block_many(engine, PROMPT_IDS)
        seconds = time.monotonic() - t0
        k3_bytes = int(engine.reader.report()["bytes_read"]) - before
        return hidden, seconds, k3_bytes, attn_stats, conv_stats, gate_stats
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
        ref_conv = ExactGDNConvSilu(args.conv_lib)
        ref_hidden, ref_seconds, ref_bytes, ref_attn_stats, ref_conv_stats, _ = _run_with_patches(
            engine, attn_core=ref_attn, conv_core=ref_conv)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        cand_attn = ExactAttentionCore(args.attn_lib)
        cand_conv = ExactGDNConvSilu(args.conv_lib)
        gate_core = ExactGDNOutputGate(args.gate_lib)
        cand_hidden, cand_seconds, cand_bytes, cand_attn_stats, cand_conv_stats, gate_stats = _run_with_patches(
            engine, attn_core=cand_attn, conv_core=cand_conv, gate_core=gate_core)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b) for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("native GDN output gate candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native GDN output gate candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("native GDN output gate A/B requires direct I/O")

        gate_report = gate_core.report()
        expected_calls = 48 * len(PROMPT_IDS)
        expected_values = expected_calls * gdn.VALUE_DIM
        if gate_report["calls"] != expected_calls or gate_report["values"] != expected_values:
            raise RuntimeError(f"unexpected native output gate coverage: {gate_report}")

        payload = {
            "schema": "qwen38-gdn-output-gate-exact-ab-v1",
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
            "native_output_gate_seconds_candidate": gate_stats["native_output_gate_seconds"],
            "native_output_gate": gate_report,
            "reference_native_conv_seconds": ref_conv_stats["native_conv_silu_seconds"],
            "candidate_native_conv_seconds": cand_conv_stats["native_conv_silu_seconds"],
            "reference_native_attention_seconds": ref_attn_stats["native_attention_core_seconds"],
            "candidate_native_attention_seconds": cand_attn_stats["native_attention_core_seconds"],
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": engine.reader.report(),
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "research_basis": [
                "FLA implements fused RMSNorm with optional norm-before SiLU gating.",
                "Liger Kernel reports operation fusion as a core technique for eliminating intermediate traffic.",
                "This candidate preserves the exact serial per-head reduction and F32 rounding contract instead of adopting parallel reductions.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_GDN_OUTPUT_GATE_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--gate-lib", type=Path, required=True)
    r = sub.add_parser("real")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--quant-lib", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--attn-lib", type=Path, required=True)
    r.add_argument("--conv-lib", type=Path, required=True)
    r.add_argument("--gate-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.gate_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
