#!/usr/bin/env python3
"""Measure Q6 persistent-pool speedup versus K3 prefetch/bind stalls.

Runs reference -> 2-worker Q6 pool -> reference on one hosted runner while
measuring main-thread K3 ``bind()`` wall time for every decoder layer.  This is
measurement-only: decoder arithmetic, Q6 row kernels, direct-I/O policy, ring
size, and async prefetch behavior are unchanged.

The goal is to test whether faster Q6 compute exposes storage latency that was
previously hidden behind per-layer compute.  Every pass remains subject to the
established hidden/state/K3/current-best coverage gates.
"""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import resource
import statistics
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_prefill_probe as block
import qwen38_q6_persistent_pool_exact_probe as q6
from fast_quantize_runtime import FastActivationQuantizer, find_base_quant_runtime
from native_f32_runtime import enable_native_f32
from q6_persistent_pool_runtime import Q6PersistentPoolRuntime
from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime
from quant_many_runtime import load_many_lib

ORDER = ("reference-a", "pool2", "reference-b")
POOL_THREADS = 2


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity() -> None:
    q6.sanity()
    if ORDER != ("reference-a", "pool2", "reference-b"):
        raise SystemExit(f"unexpected order: {ORDER}")
    if POOL_THREADS != 2:
        raise SystemExit("I/O overlap probe must isolate exactly 2 Q6 workers")
    print("QWEN38_Q6_POOL_IO_OVERLAP_SANITY PASS")


def _install_bind_timer(reader):
    original = reader.bind
    stats: dict[str, Any] = {
        "calls": 0,
        "seconds": 0.0,
        "pending_requested_calls": 0,
        "pending_requested_seconds": 0.0,
        "layers": [],
    }

    def timed_bind(layer: int):
        requested = int(layer)
        pending = getattr(reader, "_pending", None)
        pending_requested = bool(pending is not None and int(pending[0]) == requested)
        t0 = time.monotonic()
        out = original(requested)
        dt = time.monotonic() - t0
        stats["calls"] += 1
        stats["seconds"] += dt
        if pending_requested:
            stats["pending_requested_calls"] += 1
            stats["pending_requested_seconds"] += dt
        stats["layers"].append({
            "layer": requested,
            "seconds": dt,
            "pending_requested": pending_requested,
        })
        return out

    reader.bind = timed_bind
    return original, stats


def _summarize_bind(stats: dict[str, Any]) -> dict[str, Any]:
    layers = list(stats["layers"])
    ranked = sorted(layers, key=lambda x: float(x["seconds"]), reverse=True)
    return {
        "calls": int(stats["calls"]),
        "seconds": float(stats["seconds"]),
        "pending_requested_calls": int(stats["pending_requested_calls"]),
        "pending_requested_seconds": float(stats["pending_requested_seconds"]),
        "top_layer_stalls": ranked[:12],
    }


def _means(records: list[dict[str, Any]], mode: str) -> dict[str, float]:
    rows = [r for r in records if r["mode"] == mode]
    return {
        "seconds_mean": statistics.fmean(float(r["seconds"]) for r in rows),
        "q6_boundary_seconds_mean": statistics.fmean(float(r["q6_boundary_seconds"]) for r in rows),
        "bind_seconds_mean": statistics.fmean(float(r["bind"]["seconds"]) for r in rows),
        "non_q6_seconds_mean": statistics.fmean(float(r["non_q6_seconds"]) for r in rows),
        "residual_excluding_bind_seconds_mean": statistics.fmean(
            float(r["residual_excluding_bind_seconds"]) for r in rows),
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

        first_hidden: list[list[float]] | None = None
        first_final = None
        records: list[dict[str, Any]] = []

        for index, label in enumerate(ORDER):
            if index:
                pair.restore_state(engine, initial)

            active_fast = FastActivationQuantizer(base_quant)
            active_fast.install()
            if label == "pool2":
                active_pool = Q6PersistentPoolRuntime(
                    base_runtime, args.pool_lib,
                    threads=POOL_THREADS, max_rows=17408, max_vec=len(q6.PROMPT_IDS))
                runtime = active_pool
                mode = "pool2"
            else:
                runtime = FastMarshalQuantManyRuntime(
                    base_runtime, load_many_lib(args.many_lib))
                mode = "reference"

            original_bind, bind_stats = _install_bind_timer(engine.reader)
            try:
                rec = q6._run_stack(engine, args, runtime)
            finally:
                engine.reader.bind = original_bind

            final_state = pair.capture_state(engine)
            hidden_sha = block._digest_hidden_rows(rec["hidden"])
            state_sha = pair.snapshot_digest(final_state)
            fast_report = active_fast.report()
            q6._check_common(label, rec, fast_report)
            if hidden_sha != q6.KNOWN_HIDDEN_SHA256:
                raise RuntimeError(f"{label}: hidden anchor changed {hidden_sha}")
            if state_sha != q6.KNOWN_STATE_SHA256:
                raise RuntimeError(f"{label}: state anchor changed {state_sha}")

            hidden_exact = True
            state_exact = True
            state_mismatch = None
            if first_hidden is None:
                first_hidden = rec["hidden"]
                first_final = final_state
            else:
                hidden_exact = len(first_hidden) == len(rec["hidden"]) and all(
                    _f32_bytes(a) == _f32_bytes(b)
                    for a, b in zip(first_hidden, rec["hidden"])
                )
                state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, first_final)
                if not hidden_exact:
                    raise RuntimeError(f"{label}: hidden vectors differ from first reference")
                if not state_exact:
                    raise RuntimeError(f"{label}: state differs from first reference {state_mismatch}")

            bind = _summarize_bind(bind_stats)
            if bind["calls"] != 64:
                raise RuntimeError(f"{label}: expected 64 layer binds, got {bind['calls']}")
            total = float(rec["seconds"])
            q6_seconds = float(rec["q6_timing"]["seconds"])
            non_q6 = total - q6_seconds
            residual_excluding_bind = non_q6 - float(bind["seconds"])
            if residual_excluding_bind < -0.05:
                raise RuntimeError(
                    f"{label}: negative residual excluding bind {residual_excluding_bind}")

            pool_report = None
            if active_pool is not None:
                pool_report = active_pool.pool_report()
                if int(pool_report["threads"]) != POOL_THREADS:
                    raise RuntimeError(f"{label}: wrong pool thread report {pool_report}")

            records.append({
                "label": label,
                "mode": mode,
                "sequence_index": index,
                "seconds": total,
                "q6_boundary_seconds": q6_seconds,
                "non_q6_seconds": non_q6,
                "bind": bind,
                "residual_excluding_bind_seconds": residual_excluding_bind,
                "k3_bytes": int(rec["k3_bytes"]),
                "hidden_vectors_bitwise_exact": hidden_exact,
                "persistent_state_bitwise_exact": state_exact,
                "state_mismatch": state_mismatch,
                "hidden_sha256": hidden_sha,
                "state_sha256": state_sha,
                "fast_quantize": fast_report,
                "quant_many": rec["many"],
                "q6_boundary": rec["q6_timing"],
                "pool": pool_report,
                "native_seconds": rec["native_stats"],
            })

            if active_pool is not None:
                active_pool.close()
                active_pool = None
            active_fast.restore()
            active_fast = None

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("Q6 I/O overlap probe requires direct I/O")
        for key, expected in q6.EXPECTED_READER.items():
            if int(reader.get(key, -1)) != int(expected):
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")
        expected_read = q6.K3_STREAM_BYTES * len(ORDER)
        if int(reader.get("bytes_read", -1)) != expected_read:
            raise RuntimeError(
                f"reader bytes_read={reader.get('bytes_read')} expected={expected_read}")

        ref = _means(records, "reference")
        pool2 = _means(records, "pool2")
        bind_increase = pool2["bind_seconds_mean"] - ref["bind_seconds_mean"]
        non_q6_increase = pool2["non_q6_seconds_mean"] - ref["non_q6_seconds_mean"]
        explained_fraction = None
        if non_q6_increase > 0:
            explained_fraction = bind_increase / non_q6_increase

        payload = {
            "schema": "qwen38-q6-pool-io-overlap-v1",
            "status": "PASS",
            "claim": "measurement-only exact reference-pool2-reference probe of main-thread K3 bind stalls after Q6 parallelization",
            "model_sha256": gdn.SHA256,
            "prompt_token_count": len(q6.PROMPT_IDS),
            "order": list(ORDER),
            "records": records,
            "reference_mean": ref,
            "pool2_mean": pool2,
            "speedup_total": ref["seconds_mean"] / pool2["seconds_mean"],
            "speedup_q6_boundary": ref["q6_boundary_seconds_mean"] / pool2["q6_boundary_seconds_mean"],
            "bind_stall_increase_seconds": bind_increase,
            "non_q6_increase_seconds": non_q6_increase,
            "bind_fraction_of_non_q6_increase": explained_fraction,
            "reader": reader,
            "native_f32": native_f32.report(),
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "measurement_only": True,
                "q6_only": True,
                "q8_change": False,
                "reader_policy_change": False,
                "ring_change": False,
                "arithmetic_change": False,
                "threads": POOL_THREADS,
            },
            "baseline_repeat_abba_run": 33853180708,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_Q6_POOL_IO_OVERLAP_REAL_BITWISE_PASS")
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
        "model", "quant-lib", "many-lib", "pool-lib", "state-lib",
        "batch-state-lib", "f32-lib", "attn-lib", "conv-lib", "gate-lib",
        "swiglu-lib", "rmsnorm-lib", "rmsnorm-many-lib",
        "head-rmsnorm-many-lib", "residual-lib", "attention-gate-lib",
        "repeat-lib", "inventory", "work-dir", "output",
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
