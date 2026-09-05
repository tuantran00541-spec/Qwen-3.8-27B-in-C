#!/usr/bin/env python3
"""Exact layer-major multi-token prefill probe for the Qwen3.8 SSD runtime.

The current generator ingests a prompt token-major: each prompt token streams all
64 K3 layers from SSD.  This isolated research path instead keeps one layer
bound while processing every prompt token in causal order, then moves to the
next layer.  Recurrent GDN state, convolution history and full-attention KV are
still updated strictly token-by-token, so this is a scheduling change only.

The real gate compares all final-layer prompt hidden vectors and the complete
persistent decoder state against ordinary sequential ``engine.step()`` calls.
No approximate/chunkwise WY math is used here.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
import qwen38_k3_pair_reuse_probe as pair

N_LAYER = gen.N_LAYER
K3_STREAM_BYTES = 21_127_430_144


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", (float(x) for x in values)).tobytes()


def _digest_hidden_rows(rows: Sequence[Sequence[float]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        raw = _f32_bytes(row)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def _append_conv(engine: gen.StatefulK3Generator, il: int, qkv: Sequence[float]) -> None:
    hist = engine.conv_history[il]
    hist.append(array("f", qkv))
    if len(hist) > 3:
        del hist[0]


def step_block(engine: gen.StatefulK3Generator, token_ids: Sequence[int]) -> list[list[float]]:
    """Process consecutive prompt positions with one K3 bind/read per layer.

    Arithmetic inside each token/layer call is intentionally unchanged.  The
    only transformation is loop interchange:

        token-major baseline: token -> layer
        block prefill:         layer -> token

    Causal state within each layer is still mutated in token order.
    """
    ids = [int(x) for x in token_ids]
    if not ids:
        raise ValueError("token block must be non-empty")

    hidden = [gdn._embedding_row(engine.model, engine.directory, tok) for tok in ids]
    pos0 = int(engine.position)

    for il in range(N_LAYER):
        bound = engine.reader.bind(il)
        try:
            if il + 1 < N_LAYER:
                engine.reader.prefetch(il + 1)
            metas = base._layer_meta(engine.manifest, il)
            prefix = f"blk.{il}"

            def view(suffix: str):
                return engine.reader.tensor_view(bound, f"{prefix}.{suffix}")

            def vec(suffix: str):
                return gdn.f32_vector(view(suffix))

            if il % 4 == 3:
                cache = engine.caches[il]
                for j in range(len(ids)):
                    hidden[j] = gen.full_attn_step(
                        engine.runtime,
                        cache,
                        view,
                        metas,
                        vec,
                        hidden[j],
                        il,
                        pos0 + j,
                    )
            else:
                state = engine.states[il]
                hist = engine.conv_history[il]
                for j in range(len(ids)):
                    hidden[j], qkv = gen.recurrent_step(
                        engine.runtime,
                        engine.state_lib,
                        state,
                        hist,
                        view,
                        metas,
                        vec,
                        hidden[j],
                        il,
                    )
                    _append_conv(engine, il, qkv)
        finally:
            bound.release()

    engine.position += len(ids)
    return hidden


def run(args) -> dict[str, Any]:
    tokenizer = gen.load_tokenizer(args.tokenizer_json)
    rendered, prompt_ids = gen.encode_prompt(tokenizer, args.prompt, raw=args.raw_prompt)
    if not prompt_ids or len(prompt_ids) > args.max_prompt_tokens:
        raise RuntimeError(f"prompt token count {len(prompt_ids)} outside 1..{args.max_prompt_tokens}")
    if len(prompt_ids) < 2:
        raise RuntimeError("prefill block probe requires at least two prompt tokens")

    engine = gen.StatefulK3Generator(
        args.model, args.native_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        initial = pair.capture_state(engine)
        reader0 = int(engine.reader.report()["bytes_read"])

        # Reference: current token-major prompt ingest.
        seq_started = time.monotonic()
        seq_hidden: list[list[float]] = []
        for tok in prompt_ids:
            seq_hidden.append(engine.step(int(tok)))
        seq_seconds = time.monotonic() - seq_started
        seq_bytes = int(engine.reader.report()["bytes_read"]) - reader0
        seq_final = pair.capture_state(engine)
        seq_digest = _digest_hidden_rows(seq_hidden)

        # Restore semantic state only; keep reader counters cumulative so both
        # physical K3 byte deltas are directly observable on the same process.
        pair.restore_state(engine, initial)
        block_reader0 = int(engine.reader.report()["bytes_read"])
        block_started = time.monotonic()
        block_hidden = step_block(engine, prompt_ids)
        block_seconds = time.monotonic() - block_started
        block_bytes = int(engine.reader.report()["bytes_read"]) - block_reader0
        block_digest = _digest_hidden_rows(block_hidden)

        hidden_exact = len(seq_hidden) == len(block_hidden) and all(
            _f32_bytes(a) == _f32_bytes(b) for a, b in zip(seq_hidden, block_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, seq_final)
        ratio_exact = seq_bytes == len(prompt_ids) * block_bytes
        expected_seq_bytes = len(prompt_ids) * K3_STREAM_BYTES
        expected_block_bytes = K3_STREAM_BYTES

        if not hidden_exact:
            raise RuntimeError("block prefill final-layer prompt hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"block prefill persistent state mismatch: {state_mismatch}")
        if seq_bytes != expected_seq_bytes:
            raise RuntimeError(f"unexpected sequential K3 bytes: {seq_bytes} != {expected_seq_bytes}")
        if block_bytes != expected_block_bytes:
            raise RuntimeError(f"unexpected block K3 bytes: {block_bytes} != {expected_block_bytes}")
        if not ratio_exact:
            raise RuntimeError(f"K3 reuse ratio is not exact {len(prompt_ids)}:1")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("prompt block probe requires direct I/O evidence")

        payload = {
            "schema": "qwen38-k3-prompt-block-prefill-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt": args.prompt,
            "rendered_prompt": rendered,
            "raw_prompt": bool(args.raw_prompt),
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "schedule": "layer-major; token order preserved within each layer",
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "sequential_hidden_sha256": seq_digest,
            "block_hidden_sha256": block_digest,
            "sequential_final_state_sha256": pair.snapshot_digest(seq_final),
            "block_final_state_sha256": pair.snapshot_digest(pair.capture_state(engine)),
            "sequential_seconds": seq_seconds,
            "block_seconds": block_seconds,
            "speedup_vs_sequential_prefill": seq_seconds / block_seconds,
            "sequential_k3_bytes": seq_bytes,
            "block_k3_bytes": block_bytes,
            "k3_bytes_saved": seq_bytes - block_bytes,
            "k3_stream_reuse_ratio": seq_bytes / block_bytes,
            "exact_n_to_one_k3_bytes": ratio_exact,
            "reader": reader,
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "prompt": args.prompt,
            "prompt_token_count": len(prompt_ids),
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "sequential_seconds": seq_seconds,
            "block_seconds": block_seconds,
            "speedup_vs_sequential_prefill": payload["speedup_vs_sequential_prefill"],
            "sequential_k3_bytes": seq_bytes,
            "block_k3_bytes": block_bytes,
            "k3_bytes_saved": payload["k3_bytes_saved"],
            "k3_stream_reuse_ratio": payload["k3_stream_reuse_ratio"],
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2, ensure_ascii=False))
        print("QWEN38_K3_PROMPT_BLOCK_PREFILL_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert N_LAYER == 64
    assert K3_STREAM_BYTES == 21_127_430_144
    print(json.dumps({
        "schema": "qwen38-k3-prompt-block-prefill-sanity-v1",
        "status": "PASS",
        "schedule": "layer-major; causal token order preserved",
        "approximate_chunkwise_math": False,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--native-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--tokenizer-json", type=Path, required=True)
    r.add_argument("--prompt", default="Write a 700-word essay on curiosity.")
    r.add_argument("--raw-prompt", action="store_true")
    r.add_argument("--max-prompt-tokens", type=int, default=16)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
