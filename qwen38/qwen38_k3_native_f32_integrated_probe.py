#!/usr/bin/env python3
"""Integrated exact gate for reusable native-F32 + layer-major prefill.

This is intentionally not another algorithmic A/B.  The same algorithm already
passed a real same-run comparison (block-v1 236.911 s -> native-F32 198.114 s,
with exact hidden/state parity).  This gate verifies that the promoted reusable
``native_f32_runtime`` module can be installed on the normal StatefulK3Generator
and drive the proven layer-major prefill path without changing semantics,
traffic, or the known prompt-state anchors.
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
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_prefill_probe as block
from native_f32_runtime import enable_native_f32

KNOWN_PROMPT_IDS = [7734, 264, 220, 22, 15, 15, 36093, 8627, 383, 38896, 13]
KNOWN_HIDDEN_SHA256 = "e40dfb2d14456006608b095dd0c6bd018cdeed4214fdc573c8e352fb463f2e04"
KNOWN_STATE_SHA256 = "41f6fcd8f9947833956aaad0175da197456a3e678e0e31b40c5d7a08560fda06"
K3_STREAM_BYTES = 21_127_430_144


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.native_lib, args.state_lib, args.inventory, args.work_dir,
    )
    started = time.monotonic()
    try:
        runtime = enable_native_f32(engine, args.f32_lib)
        reader0 = int(engine.reader.report()["bytes_read"])
        prefill_started = time.monotonic()
        hidden = block.step_block(engine, KNOWN_PROMPT_IDS)
        prefill_seconds = time.monotonic() - prefill_started
        k3_bytes = int(engine.reader.report()["bytes_read"]) - reader0
        state = pair.capture_state(engine)
        hidden_sha = block._digest_hidden_rows(hidden)
        state_sha = pair.snapshot_digest(state)
        report = runtime.report()
        reader = engine.reader.report()

        if hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"integrated hidden digest mismatch: {hidden_sha}")
        if state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"integrated state digest mismatch: {state_sha}")
        if k3_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"integrated K3 bytes changed: {k3_bytes}")
        if not bool(reader.get("direct_io")):
            raise RuntimeError("integrated native-F32 gate requires direct I/O")
        if report["native_f32_calls"] != 1056:
            raise RuntimeError(
                f"unexpected native F32 call count: {report['native_f32_calls']}"
            )
        if report["native_f32_terms"] != 259_522_560:
            raise RuntimeError(
                f"unexpected native F32 term count: {report['native_f32_terms']}"
            )

        payload = {
            "schema": "qwen38-k3-native-f32-integrated-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": KNOWN_PROMPT_IDS,
            "prompt_token_count": len(KNOWN_PROMPT_IDS),
            "schedule": "layer-major; causal token order preserved",
            "native_f32_module": "native_f32_runtime",
            "hidden_sha256": hidden_sha,
            "state_sha256": state_sha,
            "k3_bytes": k3_bytes,
            "prefill_seconds": prefill_seconds,
            **report,
            "reader": reader,
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "prefill_seconds": prefill_seconds,
            "k3_bytes": k3_bytes,
            "native_f32_calls": report["native_f32_calls"],
            "native_f32_terms": report["native_f32_terms"],
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_K3_NATIVE_F32_INTEGRATED_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert len(KNOWN_PROMPT_IDS) == 11
    assert K3_STREAM_BYTES == 21_127_430_144
    assert len(KNOWN_HIDDEN_SHA256) == 64
    assert len(KNOWN_STATE_SHA256) == 64
    print(json.dumps({
        "schema": "qwen38-k3-native-f32-integrated-sanity-v1",
        "status": "PASS",
        "runtime": "normal StatefulK3Generator + reusable native F32 wrapper",
        "prefill": "proven layer-major step_block",
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--native-lib", type=Path, required=True)
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
