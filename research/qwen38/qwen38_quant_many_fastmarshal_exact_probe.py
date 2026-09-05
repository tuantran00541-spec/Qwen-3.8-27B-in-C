#!/usr/bin/env python3
"""Exact bulk C->Python matvec output marshaling A/B on current-best Qwen3.8.

Both sides use the same Q8 allocation-hoist C bridge and all current-best exact
native helpers. Candidate arithmetic and C matvec calls are unchanged; only the
flat ctypes c_float output conversion becomes one bulk F32 byte copy followed
by row slicing instead of Python float(...) per element.
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
import qwen38_gdn_repeat_scale_exact_probe as repeat_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from gdn_repeat_scale_runtime import ExactGDNRepeatScale
from native_f32_runtime import enable_native_f32
from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime, marshal_output_bulk
from quant_many_runtime import QuantManyRuntime, load_many_lib

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_MANY = {
    "many_calls": 400,
    "many_vectors": 4400,
    "many_rows": 42893312,
}
EXPECTED_REPEAT = {
    "calls": 48,
    "rows": 48 * len(PROMPT_IDS),
    "q_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "k_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def _finite_f32_from_bits(bits: int) -> float:
    value = struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("fixture requested non-finite F32")
    return value


def sanity() -> None:
    patterns = [
        0x00000000, 0x80000000, 0x00000001, 0x007FFFFF,
        0x00800000, 0x3F800000, 0xBF800000, 0x3EAAAAAB,
        0x7F7FFFFF, 0xFF7FFFFF,
    ]
    values = [_finite_f32_from_bits(x) for x in patterns]
    values.extend(float(i - 37) / 11.0 for i in range(127))
    rows = len(values)
    n_vec = 5
    out = (ctypes.c_float * (rows * n_vec))()
    for v in range(n_vec):
        for r in range(rows):
            out[v * rows + r] = values[(r + v * 7) % rows]

    ref = [
        [float(out[v * rows + r]) for r in range(rows)]
        for v in range(n_vec)
    ]
    cand = marshal_output_bulk(out, rows, n_vec)
    if len(ref) != len(cand) or any(
        _f32_bytes(a) != _f32_bytes(b) for a, b in zip(ref, cand)
    ):
        raise SystemExit("bulk matvec output marshal is not F32 bitwise exact")
    print("QWEN38_QUANT_MANY_FASTMARSHAL_EXACT_SANITY PASS")


def _run_current_best(engine, args, repeat_core: ExactGDNRepeatScale):
    return repeat_probe._current_best_run(engine, args, repeat_core=repeat_core)


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        base_runtime = engine.runtime
        initial = pair.capture_state(engine)

        ref_many = QuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = ref_many
        ref_repeat = ExactGDNRepeatScale(args.repeat_lib)
        (
            ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms,
            ref_residual, ref_attn_gate, ref_repeat_report,
        ) = _run_current_best(engine, args, ref_repeat)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)
        ref_many_report = ref_many.report()

        pair.restore_state(engine, initial)
        cand_many = FastMarshalQuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = cand_many
        cand_repeat = ExactGDNRepeatScale(args.repeat_lib)
        (
            cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms,
            cand_residual, cand_attn_gate, cand_repeat_report,
        ) = _run_current_best(engine, args, cand_repeat)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)
        cand_many_report = cand_many.report()

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("fast-marshal candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"fast-marshal candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        if ref_many_report != EXPECTED_MANY or cand_many_report != EXPECTED_MANY:
            raise RuntimeError(
                f"unexpected quant-many coverage ref={ref_many_report} cand={cand_many_report}")
        if ref_repeat_report != EXPECTED_REPEAT or cand_repeat_report != EXPECTED_REPEAT:
            raise RuntimeError(
                f"unexpected repeat coverage ref={ref_repeat_report} cand={cand_repeat_report}")
        if ref_rms.get("total_rows") != 6336 or cand_rms.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected RMSNorm coverage ref={ref_rms} cand={cand_rms}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("fast-marshal A/B requires direct I/O")

        payload = {
            "schema": "qwen38-quant-many-fastmarshal-exact-ab-v1",
            "status": "PASS",
            "claim": "exact bulk C-to-Python F32 matvec output marshal; same-run real GGUF A/B on GDN-repeat current-best path",
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
            "reference_quant_many": ref_many_report,
            "candidate_quant_many": cand_many_report,
            "reference_gdn_repeat_scale": ref_repeat_report,
            "candidate_gdn_repeat_scale": cand_repeat_report,
            "reference_rmsnorm": ref_rms,
            "candidate_rmsnorm": cand_rms,
            "native_f32": native_f32.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "output_f32_values_per_prefill": EXPECTED_MANY["many_rows"],
                "reference": "Python float(ctypes_c_float) once per matvec_many output value",
                "candidate": "bulk memoryview bytes -> array('f') -> row tolist",
                "c_matvec_change": False,
                "arithmetic_change": False,
            },
            "baseline_gdn_repeat_ab_run": 33840198523,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_QUANT_MANY_FASTMARSHAL_REAL_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib", "residual-lib",
        "attention-gate-lib", "repeat-lib", "inventory", "work-dir", "output",
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
