#!/usr/bin/env python3
"""Exact staged multi-vector prefill gate for Qwen3.8-27B.

This is the next step after two independent real gates:
  * layer-major prefill: one K3 weight stream for the entire prompt;
  * real layer-0 matvec_many: Q8/Q6 outputs bitwise identical with 1.62x/1.81x
    projection-level speedups for the 11-token essay prompt.

The candidate batches only state-independent matrix projections.  All causal
operations remain token-serial:
  * Gated DeltaNet convolution history and recurrent state update;
  * full-attention K/V append, softmax and value accumulation.

For each token, each quantized dot-product keeps the proven single-vector
accumulation order; only weight unpack/traversal is shared across prompt vectors.
"""
from __future__ import annotations

import argparse
from array import array
import ctypes
import json
import math
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_prefill_probe as block
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many

PROMPT_IDS = [7734, 264, 220, 22, 15, 15, 36093, 8627, 383, 38896, 13]
KNOWN_HIDDEN_SHA256 = "e40dfb2d14456006608b095dd0c6bd018cdeed4214fdc573c8e352fb463f2e04"
KNOWN_STATE_SHA256 = "41f6fcd8f9947833956aaad0175da197456a3e678e0e31b40c5d7a08560fda06"
K3_STREAM_BYTES = 21_127_430_144


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _ffn_many(runtime, view, metas, prefix: str, xs: Sequence[Sequence[float]]):
    prepared = runtime.prepare_many(xs, "Q6_K")
    gate = runtime.matvec_many(
        view("ffn_gate.weight"), metas[f"{prefix}.ffn_gate.weight"], xs, prepared=prepared)
    up = runtime.matvec_many(
        view("ffn_up.weight"), metas[f"{prefix}.ffn_up.weight"], xs, prepared=prepared)
    sw = [
        [gen.mulf(gen.t2.siluf(gate[j][i]), up[j][i]) for i in range(gdn.INTERMEDIATE)]
        for j in range(len(xs))
    ]
    return runtime.matvec_many(
        view("ffn_down.weight"), metas[f"{prefix}.ffn_down.weight"], sw)


def _recurrent_layer_many(engine, hidden, il: int, view, metas, vec):
    runtime = engine.runtime
    p = f"blk.{il}"
    attn_norm_w = vec("attn_norm.weight")
    xs = [gdn.rms_norm(h, attn_norm_w) for h in hidden]

    qkv = runtime.matvec_many(view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], xs)
    z = runtime.matvec_many(view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], xs)
    # F32 matrices use the promoted exact native fsum path.  They are small;
    # batching their weight traversal is deliberately deferred.
    beta_raw = runtime.matvec_many(view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], xs)
    alpha = runtime.matvec_many(view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], xs)

    dt = vec("ssm_dt.bias")
    aa = vec("ssm_a")
    kernels = vec("ssm_conv1d.weight")
    norm_w = vec("ssm_norm.weight")
    state = engine.states[il]
    hist = engine.conv_history[il]
    gated_rows: list[list[float]] = []

    # This is the causal barrier.  Only this core remains token-serial.
    for j in range(len(hidden)):
        beta = [gen.exact.sigmoid_f32(v) for v in beta_raw[j]]
        gate = [
            gen.mulf(aa[h], gen.t2.softplusf(gen.addf(alpha[j][h], dt[h])))
            for h in range(gdn.V_HEADS)
        ]

        prior = list(hist[-3:])
        conv = [0.0] * gdn.CONV_DIM
        for c in range(gdn.CONV_DIM):
            cur = gen.mulf(qkv[j][c], kernels[c * gdn.CONV_KERNEL + 3])
            for lag, old in enumerate(reversed(prior), start=1):
                cur = gen.addf(
                    cur,
                    gen.mulf(old[c], kernels[c * gdn.CONV_KERNEL + 3 - lag]),
                )
            conv[c] = gen.t2.siluf(cur)

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
        core = [float(out_buf[i]) for i in range(gdn.VALUE_DIM)]
        core_h = gdn.split_heads(core, gdn.V_HEADS)
        z_h = gdn.split_heads(z[j], gdn.V_HEADS)
        gated: list[float] = []
        for ch, zh in zip(core_h, z_h):
            nh = gen.rmswrap.ggml_rms_norm(ch, norm_w, gdn.RMS_EPS)
            gated.extend(
                gen.mulf(nh[d], gen.t2.siluf(zh[d]))
                for d in range(gdn.HEAD_DIM)
            )
        gated_rows.append(gated)

        # qkv is the only convolution history payload.  Updating it here is
        # equivalent to the old token-complete boundary because no later
        # operation for token j mutates recurrent state/history.
        hist.append(array("f", qkv[j]))
        if len(hist) > 3:
            del hist[0]

    linear = runtime.matvec_many(
        view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gated_rows)
    residual = [
        [gen.addf(hidden[j][i], linear[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]
    post_norm_w = vec("post_attention_norm.weight")
    post = [gdn.rms_norm(r, post_norm_w) for r in residual]
    fo = _ffn_many(runtime, view, metas, p, post)
    return [
        [gen.addf(residual[j][i], fo[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]


def _full_attention_layer_many(engine, hidden, il: int, pos0: int, view, metas, vec):
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

    # Causal attention core: K/V append and reads remain token-serial.
    for j in range(len(hidden)):
        q, gate = gen.attn.split_q_gate(qg_rows[j])
        q = gen.attn.rms_norm_heads(q, gen.attn.N_HEAD, q_norm_w)
        k = gen.attn.rms_norm_heads(k_rows[j], gen.attn.N_HEAD_KV, k_norm_w)
        pos = pos0 + j
        q_rope = gen.t2.rope_text_neox(q, gen.attn.N_HEAD, pos)
        k_rope = gen.t2.rope_text_neox(k, gen.attn.N_HEAD_KV, pos)
        cache["k"].append(gen.attn.f16_roundtrip(k_rope))
        cache["v"].append(gen.attn.f16_roundtrip(v_rows[j]))

        qh = gen.attn.split_heads(q_rope, gen.attn.N_HEAD)
        pregate: list[float] = []
        n_ctx = len(cache["k"])
        for qidx in range(gen.attn.N_HEAD):
            kvh = qidx // gen.attn.GQA_REPEAT
            qv = qh[qidx]
            scores: list[float] = []
            for ti in range(n_ctx):
                kh = cache["k"][ti][
                    kvh * gen.attn.HEAD_DIM : (kvh + 1) * gen.attn.HEAD_DIM]
                score = gen.f32(
                    math.fsum(
                        float(qv[d]) * float(kh[d])
                        for d in range(gen.attn.HEAD_DIM)
                    ) * gen.t2.SCALE_ATTN
                )
                scores.append(score)
            probs = gen.softmax_many(scores)
            for d in range(gen.attn.HEAD_DIM):
                acc = gen.f32(0.0)
                for ti in range(n_ctx):
                    vv = cache["v"][ti][kvh * gen.attn.HEAD_DIM + d]
                    acc = gen.addf(acc, gen.mulf(probs[ti], vv))
                pregate.append(acc)

        gs = [gen.exact.sigmoid_f32(vv) for vv in gate]
        gated_rows.append([
            gen.mulf(pregate[i], gs[i]) for i in range(gen.attn.Q_DIM)
        ])

    ao = runtime.matvec_many(
        view("attn_output.weight"), metas[f"{p}.attn_output.weight"], gated_rows)
    residual = [
        [gen.addf(hidden[j][i], ao[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]
    post_norm_w = vec("post_attention_norm.weight")
    post = [gdn.rms_norm(r, post_norm_w) for r in residual]
    fo = _ffn_many(runtime, view, metas, p, post)
    return [
        [gen.addf(residual[j][i], fo[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]


def step_block_many(engine: gen.StatefulK3Generator, token_ids: Sequence[int]):
    ids = [int(x) for x in token_ids]
    if not ids:
        raise ValueError("token block must be non-empty")
    hidden = [gdn._embedding_row(engine.model, engine.directory, tok) for tok in ids]
    pos0 = int(engine.position)

    for il in range(gen.N_LAYER):
        bound = engine.reader.bind(il)
        try:
            if il + 1 < gen.N_LAYER:
                engine.reader.prefetch(il + 1)
            metas = base._layer_meta(engine.manifest, il)
            prefix = f"blk.{il}"

            def view(suffix: str):
                return engine.reader.tensor_view(bound, f"{prefix}.{suffix}")

            def vec(suffix: str):
                return gdn.f32_vector(view(suffix))

            if il % 4 == 3:
                hidden = _full_attention_layer_many(
                    engine, hidden, il, pos0, view, metas, vec)
            else:
                hidden = _recurrent_layer_many(engine, hidden, il, view, metas, vec)
        finally:
            bound.release()

    engine.position += len(ids)
    return hidden


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native = enable_native_f32(engine, args.f32_lib)
        initial = pair.capture_state(engine)

        # Proven integrated native-F32 layer-major path is the same-run reference.
        ref0 = int(engine.reader.report()["bytes_read"])
        t0 = time.monotonic()
        ref_hidden = block.step_block(engine, PROMPT_IDS)
        ref_s = time.monotonic() - t0
        ref_bytes = int(engine.reader.report()["bytes_read"]) - ref0
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        many = enable_quant_many(engine, args.many_lib)
        native_calls_before = native.native_f32_calls
        cand0 = int(engine.reader.report()["bytes_read"])
        t0 = time.monotonic()
        cand_hidden = step_block_many(engine, PROMPT_IDS)
        cand_s = time.monotonic() - t0
        cand_bytes = int(engine.reader.report()["bytes_read"]) - cand0
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state = pair.capture_state(engine)
        cand_state_sha = pair.snapshot_digest(cand_state)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("staged matvec_many hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"staged matvec_many state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("staged matvec_many gate requires direct I/O")

        many_report = many.report()
        payload = {
            "schema": "qwen38-k3-prompt-block-many-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "hidden_sha256": cand_hidden_sha,
            "state_sha256": cand_state_sha,
            "reference_seconds_same_run": ref_s,
            "candidate_seconds_same_run": cand_s,
            "speedup_vs_native_f32_block_same_run": ref_s / cand_s,
            "reference_k3_bytes": ref_bytes,
            "candidate_k3_bytes": cand_bytes,
            "native_f32_calls_candidate": native.native_f32_calls - native_calls_before,
            **many_report,
            "reader": reader,
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "PASS",
            "reference_seconds_same_run": ref_s,
            "candidate_seconds_same_run": cand_s,
            "speedup": payload["speedup_vs_native_f32_block_same_run"],
            "many_calls": many_report["many_calls"],
            "many_vectors": many_report["many_vectors"],
            "native_f32_calls_candidate": payload["native_f32_calls_candidate"],
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_K3_PROMPT_BLOCK_MANY_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert len(PROMPT_IDS) == 11
    assert gen.N_LAYER == 64
    assert K3_STREAM_BYTES == 21_127_430_144
    print(json.dumps({
        "schema": "qwen38-k3-prompt-block-many-sanity-v1",
        "status": "PASS",
        "batched": "state-independent quantized projections",
        "causal_core": "token-serial GDN state and attention KV/softmax",
        "strict_exact": True,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--quant-lib", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
