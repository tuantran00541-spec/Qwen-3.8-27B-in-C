#!/usr/bin/env python3
"""Real exact Q6_K persistent row-worker pool A/B on current-best Qwen3.8.

Reference is the fully proven post-head-RMSNorm-batching stack.  Candidates
change only Q6_K matvec-many dispatch and are measured with 1/2/4 total compute
threads on the same hosted runner.  Q8_0 and all decoder arithmetic outside Q6
remain unchanged.
"""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_attention_head_rmsnorm_many_exact_probe as head_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from fast_quantize_runtime import FastActivationQuantizer, find_base_quant_runtime
from gdn_repeat_scale_runtime import ExactGDNRepeatScale
from gdn_state_batch_runtime import ExactGDNStateBatch
from native_f32_runtime import enable_native_f32
from q6_persistent_pool_runtime import Q6PersistentPoolRuntime
from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime
from quant_many_runtime import load_many_lib
from rmsnorm_heads_many_runtime import ExactRMSNormHeadsMany
from rmsnorm_many_runtime import ExactRMSNormMany

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_MANY = dict(head_probe.EXPECTED_MANY)
EXPECTED_REPEAT = dict(head_probe.EXPECTED_REPEAT)
EXPECTED_BATCH = dict(head_probe.EXPECTED_BATCH)
EXPECTED_GDN_RMS_MANY = dict(head_probe.EXPECTED_GDN_RMS_MANY)
EXPECTED_ATTN_RMS_MANY = dict(head_probe.EXPECTED_ATTN_RMS_MANY)
EXPECTED_HEAD_MANY = dict(head_probe.EXPECTED_HEAD_MANY)
EXPECTED_ZERO_WRAPPER = dict(head_probe.EXPECTED_ZERO_WRAPPER)
EXPECTED_READER = dict(head_probe.EXPECTED_READER)
EXPECTED_FAST_QUANT_CALLS = int(head_probe.EXPECTED_FAST_QUANT_CALLS)
EXPECTED_FAST_QUANT_VALUES = 28_295_168
EXPECTED_Q6 = {
    "calls": 280,
    "vectors": 3080,
    "rows": 33_882_112,
}
THREAD_COUNTS = (1, 2, 4)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity() -> None:
    if len(PROMPT_IDS) != 11:
        raise SystemExit(f"unexpected prompt count: {len(PROMPT_IDS)}")
    if EXPECTED_MANY["many_calls"] != 400 or EXPECTED_MANY["many_rows"] != 42_893_312:
        raise SystemExit(f"unexpected many contract: {EXPECTED_MANY}")
    if EXPECTED_Q6["calls"] != 280 or EXPECTED_Q6["vectors"] != 3080:
        raise SystemExit(f"unexpected Q6 coverage: {EXPECTED_Q6}")
    if EXPECTED_MANY["many_rows"] - EXPECTED_Q6["rows"] != 9_011_200:
        raise SystemExit("unexpected Q8 residual row coverage")
    if tuple(THREAD_COUNTS) != (1, 2, 4):
        raise SystemExit("thread-count contract changed")
    print("QWEN38_Q6_PERSISTENT_POOL_PROBE_SANITY PASS")


def _install_q6_timer(runtime):
    original = runtime.matvec_many
    stats = {"calls": 0, "vectors": 0, "rows": 0, "seconds": 0.0}

    def timed(weights, meta, xs, prepared=None):
        if str(meta.get("type_name")) != "Q6_K":
            return original(weights, meta, xs, prepared=prepared)
        t0 = time.monotonic()
        out = original(weights, meta, xs, prepared=prepared)
        seconds = time.monotonic() - t0
        stats["calls"] += 1
        stats["vectors"] += len(xs)
        stats["rows"] += len(xs) * int(meta["shape"][1])
        stats["seconds"] += seconds
        return out

    runtime.matvec_many = timed
    return original, stats


def _run_stack(engine, args, runtime):
    engine.runtime = runtime
    original_many, q6_timing = _install_q6_timer(runtime)
    repeat_core = ExactGDNRepeatScale(args.repeat_lib)
    state_batch_core = ExactGDNStateBatch(args.batch_state_lib)
    gdn_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
    attention_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
    head_many = ExactRMSNormHeadsMany(args.head_rmsnorm_many_lib)
    try:
        result = head_probe._run_with_head_rms_many(
            engine, args, repeat_core, state_batch_core,
            gdn_rms_many, attention_rms_many, head_many)
    finally:
        runtime.matvec_many = original_many
    hidden, seconds, k3_bytes, native_stats, rms_wrapper, residual, attn_gate, repeat = result
    return {
        "hidden": hidden,
        "seconds": seconds,
        "k3_bytes": k3_bytes,
        "native_stats": native_stats,
        "rms_wrapper": rms_wrapper,
        "residual": residual,
        "attn_gate": attn_gate,
        "repeat": repeat,
        "state_batch": state_batch_core.report(),
        "gdn_rms_many": gdn_rms_many.report(),
        "attention_rms_many": attention_rms_many.report(),
        "head_many": head_many.report(),
        "many": runtime.report(),
        "q6_timing": q6_timing,
    }


def _check_common(label: str, rec: dict[str, Any], fast_report: dict[str, Any]) -> None:
    if rec["k3_bytes"] != K3_STREAM_BYTES:
        raise RuntimeError(f"{label}: unexpected K3 bytes {rec['k3_bytes']}")
    if rec["many"] != EXPECTED_MANY:
        raise RuntimeError(f"{label}: unexpected quant-many coverage {rec['many']}")
    if rec["q6_timing"]["calls"] != EXPECTED_Q6["calls"]:
        raise RuntimeError(f"{label}: unexpected timed Q6 calls {rec['q6_timing']}")
    if rec["q6_timing"]["vectors"] != EXPECTED_Q6["vectors"]:
        raise RuntimeError(f"{label}: unexpected timed Q6 vectors {rec['q6_timing']}")
    if rec["q6_timing"]["rows"] != EXPECTED_Q6["rows"]:
        raise RuntimeError(f"{label}: unexpected timed Q6 rows {rec['q6_timing']}")
    if rec["repeat"] != EXPECTED_REPEAT:
        raise RuntimeError(f"{label}: unexpected repeat coverage {rec['repeat']}")
    if rec["state_batch"] != EXPECTED_BATCH:
        raise RuntimeError(f"{label}: unexpected state-batch coverage {rec['state_batch']}")
    if rec["gdn_rms_many"] != EXPECTED_GDN_RMS_MANY:
        raise RuntimeError(f"{label}: unexpected GDN RMS coverage {rec['gdn_rms_many']}")
    if rec["attention_rms_many"] != EXPECTED_ATTN_RMS_MANY:
        raise RuntimeError(
            f"{label}: unexpected attention RMS coverage {rec['attention_rms_many']}")
    if rec["head_many"] != EXPECTED_HEAD_MANY:
        raise RuntimeError(f"{label}: unexpected head RMS coverage {rec['head_many']}")
    if rec["rms_wrapper"] != EXPECTED_ZERO_WRAPPER:
        raise RuntimeError(f"{label}: scalar RMS wrapper returned {rec['rms_wrapper']}")
    if int(fast_report["calls"]) != EXPECTED_FAST_QUANT_CALLS:
        raise RuntimeError(f"{label}: unexpected fast-quant calls {fast_report}")
    if int(fast_report["values"]) != EXPECTED_FAST_QUANT_VALUES:
        raise RuntimeError(f"{label}: unexpected fast-quant values {fast_report}")


def _public_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "seconds": rec["seconds"],
        "k3_bytes": rec["k3_bytes"],
        "q6_boundary": rec["q6_timing"],
        "native_seconds": rec["native_stats"],
        "quant_many": rec["many"],
        "gdn_state_batch": rec["state_batch"],
        "gdn_rmsnorm_many": rec["gdn_rms_many"],
        "attention_rmsnorm_many": rec["attention_rms_many"],
        "attention_head_rmsnorm_many": rec["head_many"],
        "rmsnorm_remaining": rec["rms_wrapper"],
    }


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    active_fast: FastActivationQuantizer | None = None
    active_pool: Q6PersistentPoolRuntime | None = None
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        base_runtime = engine.runtime
        base_quant = find_base_quant_runtime(base_runtime)
        initial = pair.capture_state(engine)

        active_fast = FastActivationQuantizer(base_quant)
        active_fast.install()
        reference_runtime = FastMarshalQuantManyRuntime(
            base_runtime, load_many_lib(args.many_lib))
        reference = _run_stack(engine, args, reference_runtime)
        reference_final = pair.capture_state(engine)
        reference_hidden_sha = block._digest_hidden_rows(reference["hidden"])
        reference_state_sha = pair.snapshot_digest(reference_final)
        reference_fast_report = active_fast.report()
        active_fast.restore()
        active_fast = None

        if reference_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"reference hidden anchor changed: {reference_hidden_sha}")
        if reference_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"reference state anchor changed: {reference_state_sha}")
        _check_common("reference", reference, reference_fast_report)

        candidates: list[dict[str, Any]] = []
        for threads in THREAD_COUNTS:
            pair.restore_state(engine, initial)
            active_fast = FastActivationQuantizer(base_quant)
            active_fast.install()
            active_pool = Q6PersistentPoolRuntime(
                base_runtime, args.pool_lib,
                threads=threads, max_rows=17408, max_vec=len(PROMPT_IDS))
            candidate = _run_stack(engine, args, active_pool)
            candidate_final = pair.capture_state(engine)
            candidate_hidden_sha = block._digest_hidden_rows(candidate["hidden"])
            candidate_state_sha = pair.snapshot_digest(candidate_final)
            candidate_fast_report = active_fast.report()
            pool_report = active_pool.pool_report()

            hidden_exact = len(reference["hidden"]) == len(candidate["hidden"]) and all(
                _f32_bytes(a) == _f32_bytes(b)
                for a, b in zip(reference["hidden"], candidate["hidden"])
            )
            state_exact, state_mismatch = pair.compare_current_to_snapshot(
                engine, reference_final)
            if not hidden_exact:
                raise RuntimeError(f"Q6 pool threads={threads}: hidden mismatch")
            if not state_exact:
                raise RuntimeError(
                    f"Q6 pool threads={threads}: state mismatch {state_mismatch}")
            if candidate_hidden_sha != KNOWN_HIDDEN_SHA256:
                raise RuntimeError(
                    f"Q6 pool threads={threads}: hidden anchor changed {candidate_hidden_sha}")
            if candidate_state_sha != KNOWN_STATE_SHA256:
                raise RuntimeError(
                    f"Q6 pool threads={threads}: state anchor changed {candidate_state_sha}")
            _check_common(f"candidate-{threads}", candidate, candidate_fast_report)
            if int(pool_report["threads"]) != threads:
                raise RuntimeError(f"Q6 pool thread report mismatch: {pool_report}")
            for key in ("calls", "native_calls", "vectors", "rows"):
                expected = EXPECTED_Q6["calls" if key in ("calls", "native_calls") else key]
                if int(pool_report[key]) != expected:
                    raise RuntimeError(
                        f"Q6 pool threads={threads}: {key}={pool_report[key]} expected={expected}")

            candidate_public = _public_record(candidate)
            candidate_public.update({
                "threads": threads,
                "hidden_vectors_bitwise_exact": hidden_exact,
                "persistent_state_bitwise_exact": state_exact,
                "state_mismatch": state_mismatch,
                "hidden_sha256": candidate_hidden_sha,
                "state_sha256": candidate_state_sha,
                "fast_quantize": candidate_fast_report,
                "pool": pool_report,
                "speedup_total_vs_reference": (
                    reference["seconds"] / candidate["seconds"]
                    if candidate["seconds"] else None),
                "speedup_q6_boundary_vs_reference": (
                    reference["q6_timing"]["seconds"] / candidate["q6_timing"]["seconds"]
                    if candidate["q6_timing"]["seconds"] else None),
            })
            candidates.append(candidate_public)

            active_pool.close()
            active_pool = None
            active_fast.restore()
            active_fast = None

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("Q6 persistent-pool A/B requires direct I/O")
        for key, expected in EXPECTED_READER.items():
            if int(reader.get(key, -1)) != int(expected):
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")
        expected_total_read = K3_STREAM_BYTES * (1 + len(THREAD_COUNTS))
        if int(reader.get("bytes_read", -1)) != expected_total_read:
            raise RuntimeError(
                f"reader bytes_read={reader.get('bytes_read')} expected={expected_total_read}")

        best = min(candidates, key=lambda x: float(x["seconds"]))
        payload = {
            "schema": "qwen38-q6-persistent-pool-exact-ab-v1",
            "status": "PASS",
            "claim": "Q6_K-only persistent independent-output-row worker pool; exact current-best real GGUF same-run comparison at 1/2/4 threads",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "reference": {
                **_public_record(reference),
                "hidden_sha256": reference_hidden_sha,
                "state_sha256": reference_state_sha,
                "fast_quantize": reference_fast_report,
            },
            "candidates": candidates,
            "best_candidate_threads": best["threads"],
            "best_candidate_seconds": best["seconds"],
            "best_speedup_total_vs_reference": best["speedup_total_vs_reference"],
            "best_speedup_q6_boundary_vs_reference": best["speedup_q6_boundary_vs_reference"],
            "performance_positive_any": any(
                float(c["seconds"]) < float(reference["seconds"]) for c in candidates),
            "q6_expected_coverage": EXPECTED_Q6,
            "native_f32": native_f32.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "q6_only": True,
                "q8_change": False,
                "persistent_workers": True,
                "thread_counts": list(THREAD_COUNTS),
                "row_partition_only": True,
                "within_output_reduction_order_change": False,
                "q6_many_kernel_change": False,
                "arithmetic_change": False,
                "linux_pthread_experimental_backend": True,
                "windows_production_backend_promoted": False,
            },
            "baseline_current_best_profile_run": 33850937332,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_Q6_PERSISTENT_POOL_REAL_BITWISE_PASS")
        return payload
    finally:
        if active_pool is not None:
            active_pool.close()
        if active_fast is not None:
            active_fast.restore()
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "pool-lib", "state-lib", "batch-state-lib",
        "f32-lib", "attn-lib", "conv-lib", "gate-lib", "swiglu-lib",
        "rmsnorm-lib", "rmsnorm-many-lib", "head-rmsnorm-many-lib",
        "residual-lib", "attention-gate-lib", "repeat-lib", "inventory",
        "work-dir", "output",
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
