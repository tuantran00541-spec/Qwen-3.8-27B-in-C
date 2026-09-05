#!/usr/bin/env python3
"""Real bitwise A/B gate for the exact native full-attention causal core.

Reference: the proven staged matvec-many prefill path.
Candidate: identical staged schedule/matvecs, replacing only full-attention
score/softmax/V accumulation with attention_core_exact.c.

The candidate keeps the canonical Python F16-roundtripped K/V cache unchanged.
For this probe only, the ctypes wrapper mirrors those exact cache values in an
auxiliary contiguous F32 buffer so one C call handles all query heads for a
single token/layer.  That duplicate probe cache is reported and is not yet a
production long-context storage design.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import struct
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
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


def _f32_bytes(values: Sequence[float]) -> bytes:
    return b"".join(struct.pack("<f", float(x)) for x in values)


def _full_attention_layer_native(
    engine,
    hidden,
    il: int,
    pos0: int,
    view,
    metas,
    vec,
    core: ExactAttentionCore,
    stats: dict[str, float],
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
    residual = [
        [gen.addf(hidden[j][i], ao[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]
    post_norm_w = vec("post_attention_norm.weight")
    post = [gdn.rms_norm(r, post_norm_w) for r in residual]
    fo = prefill._ffn_many(runtime, view, metas, p, post)
    return [
        [gen.addf(residual[j][i], fo[j][i]) for i in range(gdn.HIDDEN)]
        for j in range(len(hidden))
    ]


def _reference_attention(q, cache, q_heads: int, kv_heads: int, head_dim: int, scale: float):
    repeat = q_heads // kv_heads
    qh = [list(map(float, q[h * head_dim:(h + 1) * head_dim])) for h in range(q_heads)]
    out: list[float] = []
    n_ctx = len(cache["k"])
    for qidx in range(q_heads):
        kvh = qidx // repeat
        scores: list[float] = []
        for ti in range(n_ctx):
            kh = cache["k"][ti][kvh * head_dim:(kvh + 1) * head_dim]
            scores.append(gen.f32(math.fsum(
                float(qh[qidx][d]) * float(kh[d]) for d in range(head_dim)
            ) * scale))
        probs = gen.softmax_many(scores)
        for d in range(head_dim):
            acc = gen.f32(0.0)
            for ti in range(n_ctx):
                vv = cache["v"][ti][kvh * head_dim + d]
                acc = gen.addf(acc, gen.mulf(probs[ti], vv))
            out.append(acc)
    return out


def sanity(attn_lib: Path) -> None:
    gen.exact.install()
    q_heads, kv_heads, head_dim = 6, 2, 32
    scale = 1.0 / math.sqrt(head_dim)
    for n_ctx in (1, 2, 3, 7, 16, 44):
        q = [gen.f32((((i * 37 + n_ctx * 11) % 257) - 128) / 31.0) for i in range(q_heads * head_dim)]
        k_rows = [
            [gen.f32((((t * 19 + i * 13 + 5) % 251) - 125) / 29.0) for i in range(kv_heads * head_dim)]
            for t in range(n_ctx)
        ]
        v_rows = [
            [gen.f32((((t * 23 + i * 17 + 7) % 241) - 120) / 27.0) for i in range(kv_heads * head_dim)]
            for t in range(n_ctx)
        ]
        cache = {"k": k_rows, "v": v_rows}
        ref = _reference_attention(q, cache, q_heads, kv_heads, head_dim, scale)
        core = ExactAttentionCore(attn_lib)
        cand = core.compute(
            n_ctx,
            q,
            cache,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=scale,
        )
        if _f32_bytes(ref) != _f32_bytes(cand):
            raise SystemExit(f"native attention synthetic bitwise mismatch at n_ctx={n_ctx}")
    print("QWEN38_ATTENTION_CORE_EXACT_SANITY PASS")


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        initial = pair.capture_state(engine)

        ref0 = int(engine.reader.report()["bytes_read"])
        t0 = time.monotonic()
        ref_hidden = prefill.step_block_many(engine, PROMPT_IDS)
        ref_seconds = time.monotonic() - t0
        ref_bytes = int(engine.reader.report()["bytes_read"]) - ref0
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        core = ExactAttentionCore(args.attn_lib)
        stats = {"native_attention_core_seconds": 0.0}
        original_attention = prefill._full_attention_layer_many

        def candidate_attention(engine_, hidden, il, pos0, view, metas, vec):
            return _full_attention_layer_native(
                engine_, hidden, il, pos0, view, metas, vec, core, stats)

        prefill._full_attention_layer_many = candidate_attention
        try:
            cand0 = int(engine.reader.report()["bytes_read"])
            t0 = time.monotonic()
            cand_hidden = prefill.step_block_many(engine, PROMPT_IDS)
            cand_seconds = time.monotonic() - t0
            cand_bytes = int(engine.reader.report()["bytes_read"]) - cand0
        finally:
            prefill._full_attention_layer_many = original_attention

        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)
        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)

        if not hidden_exact:
            raise RuntimeError("native attention candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"native attention candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("native attention A/B gate requires direct I/O")

        expected_calls = 16 * len(PROMPT_IDS)
        expected_context_rows = 16 * (len(PROMPT_IDS) * (len(PROMPT_IDS) + 1) // 2)
        core_report = core.report()
        if core_report["calls"] != expected_calls:
            raise RuntimeError(f"native attention call count {core_report['calls']} != {expected_calls}")
        if core_report["context_rows"] != expected_context_rows:
            raise RuntimeError(
                f"native attention context rows {core_report['context_rows']} != {expected_context_rows}")

        payload = {
            "schema": "qwen38-attention-core-exact-ab-v1",
            "status": "PASS",
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
            "native_attention_core_seconds_candidate": stats["native_attention_core_seconds"],
            "native_attention_core": core_report,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "probe_note": (
                "Candidate duplicates canonical F16-rounded K/V values into a contiguous F32 probe cache; "
                "this memory layout is evidence-only and is not yet the long-context production design."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_ATTENTION_CORE_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sp = sub.add_parser("sanity")
    sp.add_argument("--attn-lib", type=Path, required=True)
    rp = sub.add_parser("real")
    rp.add_argument("--model", type=Path, required=True)
    rp.add_argument("--quant-lib", type=Path, required=True)
    rp.add_argument("--many-lib", type=Path, required=True)
    rp.add_argument("--state-lib", type=Path, required=True)
    rp.add_argument("--f32-lib", type=Path, required=True)
    rp.add_argument("--attn-lib", type=Path, required=True)
    rp.add_argument("--inventory", type=Path, required=True)
    rp.add_argument("--work-dir", type=Path, required=True)
    rp.add_argument("--output", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.attn_lib)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
