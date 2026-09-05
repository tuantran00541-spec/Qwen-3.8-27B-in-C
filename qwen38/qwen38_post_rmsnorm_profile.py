#!/usr/bin/env python3
"""Exact post-RMSNorm hotspot profile for the current best Qwen3.8 staged prefill path.

This is a thin evidence wrapper around qwen38_post_swiglu_profile. It installs
the already-proven exact scalar native RMSNorm helpers after generator init,
then delegates the same 11-token real GGUF profile so model/hash/state/K3/
direct-I/O gates stay intact. The wrapper accounts RMSNorm time separately and
recomputes the remaining Python/orchestration bucket without changing decoder
arithmetic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_post_swiglu_profile as post
from rmsnorm_runtime import ExactRMSNorm

EXPECTED_VECTOR_CALLS = 1408
EXPECTED_VECTOR_ROWS = 1408
EXPECTED_VECTOR_VALUES = 1408 * gdn.HIDDEN
EXPECTED_HEAD_CALLS = 16 * len(post.PROMPT_IDS) * 2
EXPECTED_HEAD_ROWS = 16 * len(post.PROMPT_IDS) * (gen.attn.N_HEAD + gen.attn.N_HEAD_KV)
EXPECTED_HEAD_VALUES = EXPECTED_HEAD_ROWS * gen.attn.HEAD_DIM


def sanity() -> None:
    if EXPECTED_VECTOR_ROWS + EXPECTED_HEAD_ROWS != 6336:
        raise SystemExit("post-RMSNorm coverage sanity failed")
    if EXPECTED_VECTOR_VALUES + EXPECTED_HEAD_VALUES != 8470528:
        raise SystemExit("post-RMSNorm value coverage sanity failed")
    print("QWEN38_POST_RMSNORM_PROFILE_SANITY PASS")


def run(args) -> dict[str, Any]:
    core = ExactRMSNorm(args.rmsnorm_lib)
    old_rms = gdn.rms_norm
    old_heads = gen.attn.rms_norm_heads
    old_generator = gen.StatefulK3Generator
    rms_seconds = 0.0

    def native_rms(values, weight, eps=gdn.RMS_EPS):
        nonlocal rms_seconds
        t0 = time.monotonic()
        out = core.compute(values, weight, eps)
        rms_seconds += time.monotonic() - t0
        return out

    def native_heads(values, heads, weight):
        nonlocal rms_seconds
        t0 = time.monotonic()
        out = core.compute_heads(values, heads, weight, gen.attn.RMS_EPS)
        rms_seconds += time.monotonic() - t0
        return out

    # StatefulK3Generator.__init__ calls exact.install(), which deliberately
    # restores the pinned Python arithmetic helpers. Reapply only our already-
    # proven native RMSNorm hooks immediately after construction; this avoids
    # touching the production generator or duplicating decoder code here.
    def patched_generator(*a, **kw):
        engine = old_generator(*a, **kw)
        gdn.rms_norm = native_rms
        gen.attn.rms_norm_heads = native_heads
        return engine

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
        output=args.base_output,
    )

    gen.StatefulK3Generator = patched_generator
    try:
        base = post.run(delegated)
    finally:
        gen.StatefulK3Generator = old_generator
        gdn.rms_norm = old_rms
        gen.attn.rms_norm_heads = old_heads

    report = core.report()
    expected = {
        "calls": EXPECTED_VECTOR_CALLS,
        "rows": EXPECTED_VECTOR_ROWS,
        "values": EXPECTED_VECTOR_VALUES,
        "head_calls": EXPECTED_HEAD_CALLS,
        "head_rows": EXPECTED_HEAD_ROWS,
        "head_values": EXPECTED_HEAD_VALUES,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"unexpected post-RMSNorm coverage {key}: {report}")

    old_native = float(base["total_native_helper_seconds"])
    old_unprofiled = float(base["unprofiled_python_orchestration_seconds"])
    payload = dict(base)
    payload.update({
        "schema": "qwen38-post-rmsnorm-profile-v1",
        "claim": "exact post-RMSNorm hotspot profile; current-best native path",
        "native_rmsnorm_seconds": rms_seconds,
        "native_rmsnorm": report,
        "total_native_helper_seconds": old_native + rms_seconds,
        "unprofiled_python_orchestration_seconds": old_unprofiled - rms_seconds,
        "base_profile_schema": base.get("schema"),
    })
    payload["native_seconds"] = dict(base["native_seconds"])
    payload["native_seconds"]["native_rmsnorm_seconds"] = rms_seconds

    if payload["unprofiled_python_orchestration_seconds"] < -0.05:
        raise RuntimeError("post-RMSNorm accounting became negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("QWEN38_POST_RMSNORM_PROFILE_BITWISE_PASS")
    return payload


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib", "inventory", "work-dir",
        "base-output", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
