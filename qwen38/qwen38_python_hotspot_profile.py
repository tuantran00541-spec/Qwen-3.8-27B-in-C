#!/usr/bin/env python3
"""cProfile the exact post-SwiGLU Qwen3.8 prefill path.

This is instrumentation only.  It delegates the real 11-token run to
qwen38_post_swiglu_profile.run(), so the pinned model hash, hidden/state SHA
anchors, direct-I/O requirement, and one-stream K3 byte gate remain unchanged.
The purpose is to resolve the remaining Python/orchestration bucket into
concrete functions before changing arithmetic or scheduling.
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
from typing import Any

import qwen38_post_swiglu_profile as post


def _rows(stats: pstats.Stats, *, qwen_only: bool, sort_key: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (filename, line, funcname), (primitive_calls, total_calls, self_s, cumulative_s, _callers) in stats.stats.items():
        norm = filename.replace("\\", "/")
        if qwen_only and "/research/qwen38/" not in norm and not norm.startswith("research/qwen38/"):
            continue
        rows.append({
            "file": norm,
            "line": int(line),
            "function": funcname,
            "primitive_calls": int(primitive_calls),
            "total_calls": int(total_calls),
            "self_seconds": float(self_s),
            "cumulative_seconds": float(cumulative_s),
        })
    field = "self_seconds" if sort_key == "self" else "cumulative_seconds"
    rows.sort(key=lambda row: row[field], reverse=True)
    return rows[:limit]


def run(args) -> dict[str, Any]:
    delegated = argparse.Namespace(
        model=args.model,
        quant_lib=args.quant_lib,
        many_lib=args.many_lib,
        state_lib=args.state_lib,
        f32_lib=args.f32_lib,
        attn_lib=args.attn_lib,
        conv_lib=args.conv_lib,
        gate_lib=args.gate_lib,
        swiglu_lib=args.swiglu_lib,
        inventory=args.inventory,
        work_dir=args.work_dir,
        output=args.exact_output,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    exact_payload = post.run(delegated)
    profiler.disable()
    profiler.dump_stats(str(args.pstats_output))

    stats = pstats.Stats(profiler)
    payload = {
        "schema": "qwen38-python-hotspot-profile-v1",
        "status": exact_payload.get("status"),
        "claim": "instrumentation-only cProfile over the exact post-SwiGLU path",
        "model_sha256": exact_payload.get("model_sha256"),
        "prompt_token_count": exact_payload.get("prompt_token_count"),
        "hidden_sha256": exact_payload.get("hidden_sha256"),
        "state_sha256": exact_payload.get("state_sha256"),
        "k3_bytes": exact_payload.get("k3_bytes"),
        "direct_io": bool(exact_payload.get("reader", {}).get("direct_io")),
        "max_rss_gib": exact_payload.get("max_rss_gib"),
        "instrumented_prefill_seconds": exact_payload.get("prefill_seconds"),
        "note": "cProfile changes wall-clock cost; use these timings to rank Python hotspots, not as a speed benchmark.",
        "top_qwen_self_time": _rows(stats, qwen_only=True, sort_key="self", limit=80),
        "top_qwen_cumulative_time": _rows(stats, qwen_only=True, sort_key="cumulative", limit=80),
        "top_all_self_time": _rows(stats, qwen_only=False, sort_key="self", limit=40),
        "top_all_cumulative_time": _rows(stats, qwen_only=False, sort_key="cumulative", limit=40),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise RuntimeError(f"delegated exact profile did not pass: {payload['status']}")
    print("QWEN38_PYTHON_HOTSPOT_PROFILE_EXACT_PASS")
    return payload


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "inventory", "work-dir",
        "exact-output", "pstats-output", "output",
    ):
        ap.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
