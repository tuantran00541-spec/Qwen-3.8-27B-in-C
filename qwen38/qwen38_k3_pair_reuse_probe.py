#!/usr/bin/env python3
"""Exact two-position K3 weight-stream reuse probe for Qwen3.8-27B.

This is deliberately an isolated research path.  It does not change the default
StatefulK3Generator.  Starting from one already-proven MTP-accepted context, it
compares:

  sequential: step(A); step(B)       -> two full K3 decoder streams
  pair reuse: layer0(A,B), ...        -> one K3 stream, A then B per layer

Layer-local recurrent state, causal-convolution history and attention KV are
still updated strictly A -> B, so the layer-major schedule should be exactly
semantics-preserving.  The gate requires bitwise-equal hidden vectors and full
persistent decoder state, then reports the measured K3 byte reduction.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import resource
import struct
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
import qwen35_k3_two_token as t2

N_LAYER = gen.N_LAYER


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", (float(x) for x in values)).tobytes()


def _state_raw(state) -> bytes:
    return t2.ctypes.string_at(t2.ctypes.addressof(state), t2.STATE_BYTES_PER_LAYER)


def capture_state(engine: gen.StatefulK3Generator) -> dict[str, Any]:
    return {
        "position": int(engine.position),
        "states": {il: _state_raw(st) for il, st in engine.states.items()},
        "conv": {
            il: [array("f", row) for row in hist]
            for il, hist in engine.conv_history.items()
        },
        "caches": {
            il: {
                "k": [list(row) for row in cache["k"]],
                "v": [list(row) for row in cache["v"]],
            }
            for il, cache in engine.caches.items()
        },
    }


def restore_state(engine: gen.StatefulK3Generator, snap: dict[str, Any]) -> None:
    for il, raw in snap["states"].items():
        if len(raw) != t2.STATE_BYTES_PER_LAYER:
            raise RuntimeError(f"layer {il}: invalid state snapshot size")
        t2.ctypes.memmove(t2.ctypes.addressof(engine.states[il]), raw, len(raw))
    engine.conv_history = {
        il: [array("f", row) for row in hist]
        for il, hist in snap["conv"].items()
    }
    engine.caches = {
        il: {
            "k": [list(row) for row in cache["k"]],
            "v": [list(row) for row in cache["v"]],
        }
        for il, cache in snap["caches"].items()
    }
    engine.position = int(snap["position"])


def snapshot_bytes(snap: dict[str, Any]) -> int:
    total = sum(len(raw) for raw in snap["states"].values())
    for hist in snap["conv"].values():
        total += sum(len(row) * 4 for row in hist)
    for cache in snap["caches"].values():
        total += sum(len(row) * 2 for row in cache["k"])
        total += sum(len(row) * 2 for row in cache["v"])
    return total


def snapshot_digest(snap: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(struct.pack("<Q", int(snap["position"])))
    for il in sorted(snap["states"]):
        h.update(struct.pack("<I", int(il)))
        h.update(snap["states"][il])
    for il in sorted(snap["conv"]):
        hist = snap["conv"][il]
        h.update(struct.pack("<II", int(il), len(hist)))
        for row in hist:
            raw = row.tobytes()
            h.update(struct.pack("<I", len(raw)))
            h.update(raw)
    for il in sorted(snap["caches"]):
        h.update(struct.pack("<I", int(il)))
        for kind in ("k", "v"):
            rows = snap["caches"][il][kind]
            h.update(kind.encode("ascii"))
            h.update(struct.pack("<I", len(rows)))
            for row in rows:
                raw = _f32_bytes(row)
                h.update(struct.pack("<I", len(raw)))
                h.update(raw)
    return h.hexdigest()


def compare_current_to_snapshot(engine: gen.StatefulK3Generator, snap: dict[str, Any]) -> tuple[bool, str | None]:
    if int(engine.position) != int(snap["position"]):
        return False, "position"
    for il, expected in snap["states"].items():
        if _state_raw(engine.states[il]) != expected:
            return False, f"gdn_state:{il}"
    for il, expected_hist in snap["conv"].items():
        actual = engine.conv_history[il]
        if len(actual) != len(expected_hist):
            return False, f"conv_len:{il}"
        for j, (a, b) in enumerate(zip(actual, expected_hist)):
            if a.tobytes() != b.tobytes():
                return False, f"conv:{il}:{j}"
    for il, expected_cache in snap["caches"].items():
        actual_cache = engine.caches[il]
        for kind in ("k", "v"):
            arows = actual_cache[kind]
            brows = expected_cache[kind]
            if len(arows) != len(brows):
                return False, f"cache_len:{il}:{kind}"
            for j, (a, b) in enumerate(zip(arows, brows)):
                if _f32_bytes(a) != _f32_bytes(b):
                    return False, f"cache:{il}:{kind}:{j}"
    return True, None


def _append_conv(engine: gen.StatefulK3Generator, il: int, qkv: Sequence[float]) -> None:
    hist = engine.conv_history[il]
    hist.append(array("f", qkv))
    if len(hist) > 3:
        del hist[0]


def step_pair(engine: gen.StatefulK3Generator, token_a: int, token_b: int) -> tuple[list[float], list[float]]:
    """Run two consecutive positions while binding every K3 layer only once."""
    hidden_a = gdn._embedding_row(engine.model, engine.directory, int(token_a))
    hidden_b = gdn._embedding_row(engine.model, engine.directory, int(token_b))
    pos_a = int(engine.position)
    pos_b = pos_a + 1

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
                hidden_a = gen.full_attn_step(
                    engine.runtime, engine.caches[il], view, metas, vec, hidden_a, il, pos_a)
                hidden_b = gen.full_attn_step(
                    engine.runtime, engine.caches[il], view, metas, vec, hidden_b, il, pos_b)
            else:
                hidden_a, qkv_a = gen.recurrent_step(
                    engine.runtime,
                    engine.state_lib,
                    engine.states[il],
                    engine.conv_history[il],
                    view,
                    metas,
                    vec,
                    hidden_a,
                    il,
                )
                _append_conv(engine, il, qkv_a)
                hidden_b, qkv_b = gen.recurrent_step(
                    engine.runtime,
                    engine.state_lib,
                    engine.states[il],
                    engine.conv_history[il],
                    view,
                    metas,
                    vec,
                    hidden_b,
                    il,
                )
                _append_conv(engine, il, qkv_b)
        finally:
            bound.release()

    engine.position += 2
    return hidden_a, hidden_b


def run(args) -> dict[str, Any]:
    # This sequence is anchored by the independently reproduced llama.cpp MTP
    # oracle and the six-position exact survey:
    #   prompt 12675 -> target 11 -> target 353 -> accepted MTP draft 2688.
    prefix = [int(args.prompt_token), int(args.first_target_token)]
    pair = [int(args.pair_a), int(args.pair_b)]
    if prefix != [12675, 11] or pair != [353, 2688]:
        raise RuntimeError("this compact gate is intentionally pinned to the proven accepted MTP pair")

    engine = gen.StatefulK3Generator(
        args.model, args.native_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        for tok in prefix:
            engine.step(tok)
        start_snap = capture_state(engine)
        start_reader = engine.reader.report()
        start_bytes = int(start_reader["bytes_read"])

        # Reference schedule: two ordinary full decoder streams.
        seq_started = time.monotonic()
        seq_a = engine.step(pair[0])
        seq_b = engine.step(pair[1])
        seq_seconds = time.monotonic() - seq_started
        seq_reader = engine.reader.report()
        seq_bytes = int(seq_reader["bytes_read"]) - start_bytes
        seq_final = capture_state(engine)
        seq_a_bytes = _f32_bytes(seq_a)
        seq_b_bytes = _f32_bytes(seq_b)

        # Restore only semantic model state.  Weight-reader counters intentionally
        # keep accumulating so byte deltas remain directly observable.
        restore_state(engine, start_snap)
        pair_start_bytes = int(engine.reader.report()["bytes_read"])
        pair_started = time.monotonic()
        pair_a, pair_b = step_pair(engine, pair[0], pair[1])
        pair_seconds = time.monotonic() - pair_started
        pair_reader = engine.reader.report()
        pair_bytes = int(pair_reader["bytes_read"]) - pair_start_bytes

        hidden_a_exact = _f32_bytes(pair_a) == seq_a_bytes
        hidden_b_exact = _f32_bytes(pair_b) == seq_b_bytes
        state_exact, state_mismatch = compare_current_to_snapshot(engine, seq_final)

        # Verify that this really is the known accepted MTP pair using the exact
        # target LM head after A.  LM-head traffic is reported separately and is
        # not part of the K3 byte accounting above.
        verify_started = time.monotonic()
        verify_logits = engine.logits(pair_a)
        verify_lm_seconds = time.monotonic() - verify_started
        verify_top5 = base._topk(verify_logits, 5)
        verify_token = int(verify_top5[0]["token"])
        accepted = verify_token == pair[1]

        exact_bytes_half = seq_bytes == 2 * pair_bytes
        if not hidden_a_exact:
            raise RuntimeError("pair hidden A is not bitwise equal to sequential")
        if not hidden_b_exact:
            raise RuntimeError("pair hidden B is not bitwise equal to sequential")
        if not state_exact:
            raise RuntimeError(f"pair persistent state mismatch: {state_mismatch}")
        if not accepted:
            raise RuntimeError(f"oracle accepted pair regressed: verify={verify_token} expected={pair[1]}")
        if pair_bytes <= 20_000_000_000:
            raise RuntimeError(f"unexpectedly small K3 pair stream: {pair_bytes}")
        if not exact_bytes_half:
            raise RuntimeError(f"K3 byte reuse is not exact 2:1: seq={seq_bytes} pair={pair_bytes}")
        if not bool(pair_reader.get("direct_io")):
            raise RuntimeError("pair probe requires direct I/O evidence")

        payload = {
            "schema": "qwen38-k3-pair-reuse-probe-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prefix_token_ids": prefix,
            "pair_token_ids": pair,
            "target_verify_token": verify_token,
            "target_verify_top5": verify_top5,
            "mtp_pair_accepted": accepted,
            "hidden_a_bitwise_exact": hidden_a_exact,
            "hidden_b_bitwise_exact": hidden_b_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "start_state_sha256": snapshot_digest(start_snap),
            "sequential_final_state_sha256": snapshot_digest(seq_final),
            "pair_final_state_sha256": snapshot_digest(capture_state(engine)),
            "rollback_checkpoint_bytes": snapshot_bytes(start_snap),
            "sequential_seconds": seq_seconds,
            "pair_seconds": pair_seconds,
            "pair_speedup_vs_two_sequential_steps": seq_seconds / pair_seconds,
            "sequential_k3_bytes": seq_bytes,
            "pair_k3_bytes": pair_bytes,
            "k3_bytes_saved": seq_bytes - pair_bytes,
            "k3_stream_reuse_ratio": seq_bytes / pair_bytes,
            "exact_two_to_one_k3_bytes": exact_bytes_half,
            "verify_lm_head_seconds": verify_lm_seconds,
            "reader": pair_reader,
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "pair_token_ids": pair,
            "target_verify_token": verify_token,
            "mtp_pair_accepted": accepted,
            "hidden_a_bitwise_exact": hidden_a_exact,
            "hidden_b_bitwise_exact": hidden_b_exact,
            "persistent_state_bitwise_exact": state_exact,
            "rollback_checkpoint_bytes": payload["rollback_checkpoint_bytes"],
            "sequential_seconds": seq_seconds,
            "pair_seconds": pair_seconds,
            "pair_speedup_vs_two_sequential_steps": payload["pair_speedup_vs_two_sequential_steps"],
            "sequential_k3_bytes": seq_bytes,
            "pair_k3_bytes": pair_bytes,
            "k3_bytes_saved": payload["k3_bytes_saved"],
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_K3_PAIR_REUSE_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert N_LAYER == 64
    assert t2.STATE_BYTES_PER_LAYER * 48 == 150_994_944
    print(json.dumps({
        "schema": "qwen38-k3-pair-reuse-sanity-v1",
        "status": "PASS",
        "decoder_layers": N_LAYER,
        "gdn_state_bytes": t2.STATE_BYTES_PER_LAYER * 48,
        "schedule": "layer-major A-then-B",
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
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--prompt-token", type=int, default=12675)
    r.add_argument("--first-target-token", type=int, default=11)
    r.add_argument("--pair-a", type=int, default=353)
    r.add_argument("--pair-b", type=int, default=2688)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
