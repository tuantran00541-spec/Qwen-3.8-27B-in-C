#!/usr/bin/env python3
"""Exact block-prefill cache optimization probe.

The layer-major schedule is already proven bitwise exact on the real 11-token
essay prompt.  This probe keeps that schedule and removes avoidable Python
reparsing: F32 vectors and F32 matrices are decoded once per bound layer and
reused for all prompt tokens.  Quantized Q6_K/Q8_0 arithmetic is unchanged.
"""
from __future__ import annotations

import argparse
from array import array
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

K3_STREAM_BYTES = block.K3_STREAM_BYTES
KNOWN_PROMPT_IDS = [7734, 264, 220, 22, 15, 15, 36093, 8627, 383, 38896, 13]
KNOWN_HIDDEN_SHA256 = "e40dfb2d14456006608b095dd0c6bd018cdeed4214fdc573c8e352fb463f2e04"
KNOWN_STATE_SHA256 = "41f6fcd8f9947833956aaad0175da197456a3e678e0e31b40c5d7a08560fda06"


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


class LayerCachedRuntime:
    """Cache only immutable F32 matrix decoding for one bound layer."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.f32_matrices: dict[str, tuple[list[float], int, int]] = {}

    def matvec(self, weights: memoryview, meta: dict[str, Any], x: Sequence[float], prepared=None):
        if meta["type_name"] != "F32":
            return self.runtime.matvec(weights, meta, x, prepared=prepared)
        ne0, rows = map(int, meta["shape"])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: x={len(x)} ne0={ne0}")
        cached = self.f32_matrices.get(meta["name"])
        if cached is None:
            vals = gdn.f32_vector(weights)
            if len(vals) != ne0 * rows:
                raise ValueError(f"{meta['name']}: F32 matrix size mismatch")
            cached = (vals, ne0, rows)
            self.f32_matrices[meta["name"]] = cached
        vals, c_ne0, c_rows = cached
        if c_ne0 != ne0 or c_rows != rows:
            raise RuntimeError("cached F32 matrix contract changed")
        out: list[float] = []
        for row in range(rows):
            base_i = row * ne0
            out.append(math.fsum(vals[base_i + i] * float(x[i]) for i in range(ne0)))
        return out


def _append_conv(engine: gen.StatefulK3Generator, il: int, qkv: Sequence[float]) -> None:
    hist = engine.conv_history[il]
    hist.append(array("f", qkv))
    if len(hist) > 3:
        del hist[0]


def step_block_cached(engine: gen.StatefulK3Generator, token_ids: Sequence[int]) -> list[list[float]]:
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
            vec_cache: dict[str, list[float]] = {}
            runtime = LayerCachedRuntime(engine.runtime)

            def view(suffix: str):
                return engine.reader.tensor_view(bound, f"{prefix}.{suffix}")

            def vec(suffix: str):
                cached = vec_cache.get(suffix)
                if cached is None:
                    cached = gdn.f32_vector(view(suffix))
                    vec_cache[suffix] = cached
                return cached

            if il % 4 == 3:
                cache = engine.caches[il]
                for j in range(len(ids)):
                    hidden[j] = gen.full_attn_step(
                        runtime, cache, view, metas, vec, hidden[j], il, pos0 + j,
                    )
            else:
                state = engine.states[il]
                hist = engine.conv_history[il]
                for j in range(len(ids)):
                    hidden[j], qkv = gen.recurrent_step(
                        runtime, engine.state_lib, state, hist,
                        view, metas, vec, hidden[j], il,
                    )
                    _append_conv(engine, il, qkv)
        finally:
            bound.release()

    engine.position += len(ids)
    return hidden


def run(args) -> dict[str, Any]:
    tokenizer = gen.load_tokenizer(args.tokenizer_json)
    rendered, prompt_ids = gen.encode_prompt(tokenizer, args.prompt, raw=args.raw_prompt)
    if prompt_ids != KNOWN_PROMPT_IDS:
        raise RuntimeError(f"prompt token anchor changed: {prompt_ids}")

    engine = gen.StatefulK3Generator(
        args.model, args.native_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        initial = pair.capture_state(engine)
        reader0 = int(engine.reader.report()["bytes_read"])
        ref_started = time.monotonic()
        ref_hidden = block.step_block(engine, prompt_ids)
        ref_seconds = time.monotonic() - ref_started
        ref_bytes = int(engine.reader.report()["bytes_read"]) - reader0
        ref_state = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_state)

        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or ref_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("proven block-v1 anchor changed before cache comparison")

        pair.restore_state(engine, initial)
        cand_reader0 = int(engine.reader.report()["bytes_read"])
        cand_started = time.monotonic()
        cand_hidden = step_block_cached(engine, prompt_ids)
        cand_seconds = time.monotonic() - cand_started
        cand_bytes = int(engine.reader.report()["bytes_read"]) - cand_reader0
        cand_state = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_state)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b) for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_state)
        if not hidden_exact or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("cached block hidden vectors are not bitwise exact")
        if not state_exact or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"cached block state mismatch: {state_mismatch}")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cached={cand_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("cache probe requires direct I/O")

        payload = {
            "schema": "qwen38-k3-prompt-block-f32-cache-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt": args.prompt,
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "hidden_sha256": cand_hidden_sha,
            "state_sha256": cand_state_sha,
            "block_v1_seconds_same_run": ref_seconds,
            "block_cached_seconds_same_run": cand_seconds,
            "speedup_vs_block_v1_same_run": ref_seconds / cand_seconds,
            "block_v1_k3_bytes": ref_bytes,
            "block_cached_k3_bytes": cand_bytes,
            "reader": reader,
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "prompt_token_count": len(prompt_ids),
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "block_v1_seconds_same_run": ref_seconds,
            "block_cached_seconds_same_run": cand_seconds,
            "speedup_vs_block_v1_same_run": payload["speedup_vs_block_v1_same_run"],
            "block_v1_k3_bytes": ref_bytes,
            "block_cached_k3_bytes": cand_bytes,
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_K3_PROMPT_BLOCK_F32_CACHE_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert len(KNOWN_PROMPT_IDS) == 11
    assert K3_STREAM_BYTES == 21_127_430_144
    assert len(KNOWN_HIDDEN_SHA256) == 64 and len(KNOWN_STATE_SHA256) == 64
    print(json.dumps({
        "schema": "qwen38-k3-prompt-block-f32-cache-sanity-v1",
        "status": "PASS",
        "optimization": "layer-local immutable F32 decode cache only",
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
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
