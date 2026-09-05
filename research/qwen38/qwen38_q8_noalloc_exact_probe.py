#!/usr/bin/env python3
"""Exact Q8_0 matvec-many allocation-hoist A/B for Qwen3.8 staged prefill.

Reference and candidate both use the current-best exact path, including native
RMSNorm, SwiGLU, and residual additions. The only candidate difference is the
Q8_0 many-vector bridge: its per-output-row sums allocation is hoisted to one
allocation per matvec call. Q6_K arithmetic is delegated unchanged.
"""
from __future__ import annotations

import argparse
from array import array
import ctypes
import json
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
import qwen38_residual_add_exact_probe as res_probe
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many, load_many_lib
from residual_add_runtime import ExactResidualAdd

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def _q8_block(seed: int) -> bytes:
    # Keep scales finite and deterministic; all int8 payload patterns are valid.
    scale = 0.015625 * (1 + (seed % 7))
    q = bytes((((seed * 29 + i * 37) % 255) - 127) & 0xFF for i in range(32))
    return struct.pack("<e", scale) + q


def _q8_fixture(rows: int, n: int, n_vec: int) -> tuple[bytes, bytes, int]:
    if n % 32:
        raise ValueError("Q8 fixture width must be divisible by 32")
    nb = n // 32
    row_bytes = nb * 34
    weights = b"".join(
        _q8_block(1000 + r * nb + ib)
        for r in range(rows)
        for ib in range(nb)
    )
    activations = b"".join(
        _q8_block(200000 + v * nb + ib)
        for v in range(n_vec)
        for ib in range(nb)
    )
    return weights, activations, row_bytes


def _call_q8(lib, weights: bytes, activations: bytes, rows: int, n: int,
             act_bytes: int, n_vec: int) -> bytes:
    w = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
    a = (ctypes.c_uint8 * len(activations)).from_buffer_copy(activations)
    out = (ctypes.c_float * (rows * n_vec))()
    rc = lib.qwen_matvec_many_q8_0_q8_0_bridge(
        w, len(weights), rows, n, a, act_bytes, n_vec, out)
    if rc != 0:
        raise RuntimeError(f"Q8 synthetic bridge rc={rc}")
    return bytes(memoryview(out))


def sanity(baseline_lib: Path, candidate_lib: Path) -> None:
    ref_lib = load_many_lib(baseline_lib)
    cand_lib = load_many_lib(candidate_lib)
    for rows, n, n_vec in ((1, 32, 1), (3, 64, 2), (17, 256, 5), (257, 512, 11)):
        w, a, act_bytes = _q8_fixture(rows, n, n_vec)
        ref = _call_q8(ref_lib, w, a, rows, n, act_bytes, n_vec)
        cand = _call_q8(cand_lib, w, a, rows, n, act_bytes, n_vec)
        if ref != cand:
            raise SystemExit(
                f"Q8 noalloc bitwise mismatch rows={rows} n={n} n_vec={n_vec}")
    print("QWEN38_Q8_NOALLOC_EXACT_SANITY PASS")


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.baseline_many_lib)
        initial = pair.capture_state(engine)

        ref_residual = ExactResidualAdd(args.residual_lib)
        ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms = res_probe._best_run(
            engine, args, residual_core=ref_residual)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)

        pair.restore_state(engine, initial)
        many.many_lib = load_many_lib(args.candidate_many_lib)
        cand_residual = ExactResidualAdd(args.residual_lib)
        cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms = res_probe._best_run(
            engine, args, residual_core=cand_residual)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("Q8 noalloc candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"Q8 noalloc candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("Q8 noalloc A/B requires direct I/O")

        expected_residual = {"calls": 128, "rows": 1408, "values": 7208960}
        if ref_residual.report() != expected_residual or cand_residual.report() != expected_residual:
            raise RuntimeError(
                f"unexpected residual coverage ref={ref_residual.report()} cand={cand_residual.report()}")
        if ref_rms.get("total_rows") != 6336 or cand_rms.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected RMSNorm coverage ref={ref_rms} cand={cand_rms}")

        payload = {
            "schema": "qwen38-q8-noalloc-exact-ab-v1",
            "status": "PASS",
            "claim": "exact Q8_0 matvec-many allocation hoist; same-run real GGUF A/B on current-best residual path",
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
            "reference_native_seconds": ref_stats,
            "candidate_native_seconds": cand_stats,
            "reference_residual": ref_residual.report(),
            "candidate_residual": cand_residual.report(),
            "reference_rmsnorm": ref_rms,
            "candidate_rmsnorm": cand_rms,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "q8_output_rows_per_prefill": 819200,
                "baseline_allocation_pattern": "calloc/free once per Q8 output row",
                "candidate_allocation_pattern": "calloc/free once per Q8 matvec-many call; memset once per output row",
                "arithmetic_change": False,
            },
            "baseline_residual_ab_run": 33831051629,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_Q8_NOALLOC_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--baseline-many-lib", type=Path, required=True)
    s.add_argument("--candidate-many-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "baseline-many-lib", "candidate-many-lib", "state-lib",
        "f32-lib", "attn-lib", "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib",
        "residual-lib", "inventory", "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.baseline_many_lib, args.candidate_many_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
