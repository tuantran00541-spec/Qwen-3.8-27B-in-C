#!/usr/bin/env python3
"""Real exact layer-major prefill A/B for 4-thread GDN state updates.

Everything except the recurrent-state shared library is identical.  Baseline
uses the proven serial qwen_gdn_ar_step_f32; candidate exports the same ABI but
partitions Qwen3.8's 48 independent value-head state planes over four threads.
Token order, convolution history, attention KV, quantized matvecs, and Python
front-end arithmetic are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen35_k3_two_token as t2
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_prefill_probe as block

KNOWN_PROMPT_IDS = [7734, 264, 220, 22, 15, 15, 36093, 8627, 383, 38896, 13]
KNOWN_HIDDEN_SHA256 = "e40dfb2d14456006608b095dd0c6bd018cdeed4214fdc573c8e352fb463f2e04"
KNOWN_STATE_SHA256 = "41f6fcd8f9947833956aaad0175da197456a3e678e0e31b40c5d7a08560fda06"
K3_STREAM_BYTES = 21_127_430_144


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.native_lib, args.serial_state_lib, args.inventory, args.work_dir,
    )
    started = time.monotonic()
    try:
        initial = pair.capture_state(engine)

        reader0 = int(engine.reader.report()["bytes_read"])
        a_started = time.monotonic()
        ref_hidden = block.step_block(engine, KNOWN_PROMPT_IDS)
        a_seconds = time.monotonic() - a_started
        a_bytes = int(engine.reader.report()["bytes_read"]) - reader0
        ref_state = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_state)
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or ref_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("serial block digest anchor changed")
        if a_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"serial block K3 bytes changed: {a_bytes}")

        pair.restore_state(engine, initial)
        engine.state_lib = t2.load_state_lib(args.parallel_state_lib)
        reader1 = int(engine.reader.report()["bytes_read"])
        b_started = time.monotonic()
        cand_hidden = block.step_block(engine, KNOWN_PROMPT_IDS)
        b_seconds = time.monotonic() - b_started
        b_bytes = int(engine.reader.report()["bytes_read"]) - reader1
        cand_state = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_state)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_state)
        if not hidden_exact or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("4-thread GDN block hidden vectors are not bitwise exact")
        if not state_exact or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"4-thread GDN state mismatch: {state_mismatch}")
        if b_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"4-thread GDN block K3 bytes changed: {b_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("real GDN parallel probe requires direct I/O")

        payload = {
            "schema": "qwen38-k3-prompt-block-gdn-4t-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": KNOWN_PROMPT_IDS,
            "prompt_token_count": len(KNOWN_PROMPT_IDS),
            "gdn_threads": 4,
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "hidden_sha256": cand_hidden_sha,
            "state_sha256": cand_state_sha,
            "block_serial_seconds_same_run": a_seconds,
            "block_gdn_4t_seconds_same_run": b_seconds,
            "speedup_vs_serial_same_run": a_seconds / b_seconds,
            "block_serial_k3_bytes": a_bytes,
            "block_gdn_4t_k3_bytes": b_bytes,
            "reader": reader,
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "PASS",
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "block_serial_seconds_same_run": a_seconds,
            "block_gdn_4t_seconds_same_run": b_seconds,
            "speedup_vs_serial_same_run": payload["speedup_vs_serial_same_run"],
            "block_serial_k3_bytes": a_bytes,
            "block_gdn_4t_k3_bytes": b_bytes,
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_K3_PROMPT_BLOCK_GDN_4T_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert len(KNOWN_PROMPT_IDS) == 11
    assert K3_STREAM_BYTES == 21_127_430_144
    print(json.dumps({
        "schema": "qwen38-k3-prompt-block-gdn-4t-sanity-v1",
        "status": "PASS",
        "candidate_delta": "GDN state bridge only: serial -> 4 head threads",
        "token_order": "unchanged and serial",
        "scheduler": "proven block.step_block unchanged",
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--native-lib", type=Path, required=True)
    r.add_argument("--serial-state-lib", type=Path, required=True)
    r.add_argument("--parallel-state-lib", type=Path, required=True)
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
