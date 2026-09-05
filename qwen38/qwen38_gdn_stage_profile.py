#!/usr/bin/env python3
"""Exact 11-token internal stage profile for Qwen3.8 Gated DeltaNet.

This probe keeps the already-proven staged matvec-many arithmetic and native
attention core, then replaces only the Python GDN helper with an instrumented
copy of the same operations.  The final hidden/state SHA anchors must remain
bitwise identical to the proven 11-token prefill before timing evidence is
accepted.

The purpose is attribution, not optimization: identify which exact-compatible
GDN stages deserve native fusion/tiling before changing runtime arithmetic.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import ctypes
import json
from pathlib import Path
import resource
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_attention_core_exact_probe as attn_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from attention_core_runtime import ExactAttentionCore
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


class StageProfiler:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.layer_seconds: dict[int, float] = defaultdict(float)
        self.layer_calls: dict[int, int] = defaultdict(int)

    def add(self, name: str, elapsed: float) -> None:
        self.seconds[name] += float(elapsed)
        self.calls[name] += 1

    def add_layer(self, il: int, elapsed: float) -> None:
        self.layer_seconds[int(il)] += float(elapsed)
        self.layer_calls[int(il)] += 1

    def report(self) -> dict[str, Any]:
        total = sum(self.seconds.values())
        stages = []
        for name, seconds in sorted(self.seconds.items(), key=lambda kv: kv[1], reverse=True):
            stages.append({
                "stage": name,
                "seconds": float(seconds),
                "calls": int(self.calls[name]),
                "pct_of_profiled_gdn": (100.0 * float(seconds) / total) if total else 0.0,
            })
        return {
            "profiled_stage_seconds": total,
            "stages": stages,
            "layer_seconds": {str(k): float(v) for k, v in sorted(self.layer_seconds.items())},
            "layer_calls": {str(k): int(v) for k, v in sorted(self.layer_calls.items())},
        }


def _profiled_recurrent_layer_many(prof: StageProfiler, engine, hidden, il: int, view, metas, vec):
    layer_t0 = time.monotonic()
    runtime = engine.runtime
    p = f"blk.{il}"

    t0 = time.monotonic()
    attn_norm_w = vec("attn_norm.weight")
    xs = [gdn.rms_norm(h, attn_norm_w) for h in hidden]
    prof.add("input_rms_norm", time.monotonic() - t0)

    t0 = time.monotonic()
    qkv = runtime.matvec_many(view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], xs)
    z = runtime.matvec_many(view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], xs)
    beta_raw = runtime.matvec_many(view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], xs)
    alpha = runtime.matvec_many(view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], xs)
    prof.add("input_projections", time.monotonic() - t0)

    dt = vec("ssm_dt.bias")
    aa = vec("ssm_a")
    kernels = vec("ssm_conv1d.weight")
    norm_w = vec("ssm_norm.weight")
    state = engine.states[il]
    hist = engine.conv_history[il]
    gated_rows: list[list[float]] = []

    for j in range(len(hidden)):
        t0 = time.monotonic()
        beta = [gen.exact.sigmoid_f32(v) for v in beta_raw[j]]
        gate = [
            gen.mulf(aa[h], gen.t2.softplusf(gen.addf(alpha[j][h], dt[h])))
            for h in range(gdn.V_HEADS)
        ]
        prof.add("gate_beta_activation", time.monotonic() - t0)

        t0 = time.monotonic()
        prior = list(hist[-3:])
        conv = [0.0] * gdn.CONV_DIM
        for c in range(gdn.CONV_DIM):
            cur = gen.mulf(qkv[j][c], kernels[c * gdn.CONV_KERNEL + 3])
            for lag, old in enumerate(reversed(prior), start=1):
                cur = gen.addf(cur, gen.mulf(old[c], kernels[c * gdn.CONV_KERNEL + 3 - lag]))
            conv[c] = gen.t2.siluf(cur)
        prof.add("causal_conv_silu", time.monotonic() - t0)

        t0 = time.monotonic()
        q = conv[: gdn.KEY_DIM]
        k = conv[gdn.KEY_DIM : 2 * gdn.KEY_DIM]
        v = conv[2 * gdn.KEY_DIM :]
        qn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(q, gdn.K_HEADS)])
        kn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(k, gdn.K_HEADS)])
        q48 = [gen.mulf(vv, gen.t2.SCALE_GDN) for vv in gen.t2.repeat_k_heads(qn)]
        k48 = gen.t2.repeat_k_heads(kn)
        prof.add("qk_l2norm_repeat", time.monotonic() - t0)

        t0 = time.monotonic()
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
        prof.add("recurrent_state_kernel", time.monotonic() - t0)
        if rc != 0:
            raise RuntimeError(f"layer {il}: GDN state kernel rc={rc}")

        t0 = time.monotonic()
        core = [float(out_buf[i]) for i in range(gdn.VALUE_DIM)]
        prof.add("state_output_unpack", time.monotonic() - t0)

        t0 = time.monotonic()
        core_h = gdn.split_heads(core, gdn.V_HEADS)
        z_h = gdn.split_heads(z[j], gdn.V_HEADS)
        gated: list[float] = []
        for ch, zh in zip(core_h, z_h):
            nh = gen.rmswrap.ggml_rms_norm(ch, norm_w, gdn.RMS_EPS)
            gated.extend(gen.mulf(nh[d], gen.t2.siluf(zh[d])) for d in range(gdn.HEAD_DIM))
        gated_rows.append(gated)
        prof.add("output_rmsnorm_gate", time.monotonic() - t0)

        t0 = time.monotonic()
        hist.append(array("f", qkv[j]))
        if len(hist) > 3:
            del hist[0]
        prof.add("conv_history_update", time.monotonic() - t0)

    t0 = time.monotonic()
    linear = runtime.matvec_many(view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gated_rows)
    prof.add("ssm_out_projection", time.monotonic() - t0)

    t0 = time.monotonic()
    residual = [
        [gen.addf(hidden[j][i], linear[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]
    prof.add("attention_residual_add", time.monotonic() - t0)

    t0 = time.monotonic()
    post_norm_w = vec("post_attention_norm.weight")
    post = [gdn.rms_norm(r, post_norm_w) for r in residual]
    prof.add("post_attention_rms_norm", time.monotonic() - t0)

    t0 = time.monotonic()
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    prof.add("ffn_many", time.monotonic() - t0)

    t0 = time.monotonic()
    out = [
        [gen.addf(residual[j][i], fo[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]
    prof.add("ffn_residual_add", time.monotonic() - t0)
    prof.add_layer(int(il), time.monotonic() - layer_t0)
    return out


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        core = ExactAttentionCore(args.attn_lib)
        attn_stats = {"native_attention_core_seconds": 0.0}
        profiler = StageProfiler()

        original_recurrent = prefill._recurrent_layer_many
        original_attention = prefill._full_attention_layer_many

        def candidate_recurrent(engine_, hidden, il, view, metas, vec):
            return _profiled_recurrent_layer_many(profiler, engine_, hidden, il, view, metas, vec)

        def candidate_attention(engine_, hidden, il, pos0, view, metas, vec):
            return attn_probe._full_attention_layer_native(
                engine_, hidden, il, pos0, view, metas, vec, core, attn_stats)

        prefill._recurrent_layer_many = candidate_recurrent
        prefill._full_attention_layer_many = candidate_attention
        try:
            before = int(engine.reader.report()["bytes_read"])
            t0 = time.monotonic()
            hidden_rows = prefill.step_block_many(engine, PROMPT_IDS)
            prefill_seconds = time.monotonic() - t0
            k3_bytes = int(engine.reader.report()["bytes_read"]) - before
        finally:
            prefill._recurrent_layer_many = original_recurrent
            prefill._full_attention_layer_many = original_attention

        hidden_sha = block._digest_hidden_rows(hidden_rows)
        state_sha = pair.snapshot_digest(pair.capture_state(engine))
        if hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"instrumented GDN changed hidden anchor: {hidden_sha}")
        if state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"instrumented GDN changed state anchor: {state_sha}")
        if k3_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"expected one K3 stream, got {k3_bytes}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("GDN stage profile requires direct I/O")

        report = profiler.report()
        payload = {
            "schema": "qwen38-gdn-stage-profile-v1",
            "status": "PASS",
            "claim": "11-token exact anchored profile; arithmetic/order unchanged, instrumentation only",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_sha256": hidden_sha,
            "state_sha256": state_sha,
            "prefill_seconds": prefill_seconds,
            "gdn_profile": report,
            "native_attention_core_seconds": attn_stats["native_attention_core_seconds"],
            "native_attention_core": core.report(),
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "k3_bytes": k3_bytes,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "research_notes": [
                "Chunk/WY GDN algorithms intentionally excluded from strict bitwise path because they reassociate recurrence arithmetic.",
                "FLA-style fusion of q/k normalization, gate/beta activation and recurrent update remains an exact-path candidate if this profile justifies it.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "prefill_seconds": prefill_seconds,
            "gdn_profile": report,
            "native_attention_core_seconds": payload["native_attention_core_seconds"],
            "k3_bytes": k3_bytes,
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2), flush=True)
        print("QWEN38_GDN_STAGE_PROFILE_BITWISE_PASS", flush=True)
        return payload
    finally:
        engine.close()


def sanity() -> None:
    p = StageProfiler()
    p.add("a", 2.0)
    p.add("b", 1.0)
    r = p.report()
    assert r["stages"][0]["stage"] == "a"
    assert abs(r["stages"][0]["pct_of_profiled_gdn"] - 66.6666666667) < 1e-6
    print("QWEN38_GDN_STAGE_PROFILE_SANITY PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--quant-lib", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--attn-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
