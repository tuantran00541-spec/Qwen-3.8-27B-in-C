#!/usr/bin/env python3
"""Counterbalanced exact Q6_K persistent-pool repeat A/B on current-best Qwen3.8.

Runs ABBA on one hosted runner:
  reference -> 2-worker pool -> 2-worker pool -> reference

The Q6 kernel and pool implementation are unchanged from the already-proven
persistent-pool experiment.  This probe exists only to distinguish a real Q6
boundary win from temporal runner drift/thermal/I/O effects.  Every pass must
reproduce the established hidden/state anchors, one exact K3 stream, and all
current-best coverage counters.
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

ORDER = ("reference-a", "pool2-a", "pool2-b", "reference-b")
POOL_THREADS = 2


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def sanity() -> None:
    q6.sanity()
    if ORDER != ("reference-a", "pool2-a", "pool2-b", "reference-b"):
        raise SystemExit(f"unexpected ABBA order: {ORDER}")
    if POOL_THREADS != 2:
        raise SystemExit("repeat probe must isolate exactly 2 Q6 workers")
    print("QWEN38_Q6_PERSISTENT_POOL_REPEAT_ABBA_SANITY PASS")


def _public_record(label: str, rec: dict[str, Any], fast: dict[str, Any], pool=None) -> dict[str, Any]:
    total = float(rec["seconds"])
    q6_seconds = float(rec["q6_timing"]["seconds"])
    non_q6 = total - q6_seconds
    if non_q6 < -0.05:
        raise RuntimeError(f"{label}: negative non-Q6 accounting {non_q6}")
    out = {
        "label": label,
        "mode": "pool2" if label.startswith("pool2") else "reference",
        "seconds": total,
        "q6_boundary_seconds": q6_seconds,
        "non_q6_seconds": non_q6,
        "k3_bytes": int(rec["k3_bytes"]),
        "q6_boundary": rec["q6_timing"],
        "native_seconds": rec["native_stats"],
        "quant_many": rec["many"],
        "fast_quantize": fast,
        "gdn_state_batch": rec["state_batch"],
        "gdn_rmsnorm_many": rec["gdn_rms_many"],
        "attention_rmsnorm_many": rec["attention_rms_many"],
        "attention_head_rmsnorm_many": rec["head_many"],
        "rmsnorm_remaining": rec["rms_wrapper"],
    }
    if pool is not None:
        out["pool"] = pool
    return out


def _means(records: list[dict[str, Any]], mode: str) -> dict[str, float]:
    rows = [r for r in records if r["mode"] == mode]
    return {
        "seconds_mean": statistics.fmean(float(r["seconds"]) for r in rows),
        "q6_boundary_seconds_mean": statistics.fmean(float(r["q6_boundary_seconds"]) for r in rows),
        "non_q6_seconds_mean": statistics.fmean(float(r["non_q6_seconds"]) for r in rows),
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

        records: list[dict[str, Any]] = []
        first_hidden: list[list[float]] | None = None
        first_final = None

        for idx, label in enumerate(ORDER):
            if idx:
                pair.restore_state(engine, initial)

            active_fast = FastActivationQuantizer(base_quant)
            active_fast.install()
            if label.startswith("pool2"):
                active_pool = Q6PersistentPoolRuntime(
                    base_runtime, args.pool_lib,
                    threads=POOL_THREADS, max_rows=17408, max_vec=len(q6.PROMPT_IDS))
                runtime = active_pool
            else:
                runtime = FastMarshalQuantManyRuntime(
                    base_runtime, load_many_lib(args.many_lib))

            rec = q6._run_stack(engine, args, runtime)
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
                state_exact, state_mismatch = pair.compare_current_to_snapshot(
                    engine, first_final)
                if not hidden_exact:
                    raise RuntimeError(f"{label}: hidden vectors differ from first reference")
                if not state_exact:
                    raise RuntimeError(f"{label}: state differs from first reference {state_mismatch}")

            pool_report = None
            if active_pool is not None:
                pool_report = active_pool.pool_report()
                if int(pool_report["threads"]) != POOL_THREADS:
                    raise RuntimeError(f"{label}: pool thread report mismatch {pool_report}")
                for key in ("calls", "native_calls", "vectors", "rows"):
                    expected_key = "calls" if key in ("calls", "native_calls") else key
                    expected = q6.EXPECTED_Q6[expected_key]
                    if int(pool_report[key]) != int(expected):
                        raise RuntimeError(
                            f"{label}: pool {key}={pool_report[key]} expected={expected}")

            pub = _public_record(label, rec, fast_report, pool_report)
            pub.update({
                "hidden_vectors_bitwise_exact": hidden_exact,
                "persistent_state_bitwise_exact": state_exact,
                "state_mismatch": state_mismatch,
                "hidden_sha256": hidden_sha,
                "state_sha256": state_sha,
                "sequence_index": idx,
            })
            records.append(pub)

            if active_pool is not None:
                active_pool.close()
                active_pool = None
            active_fast.restore()
            active_fast = None

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("Q6 repeat ABBA requires direct I/O")
        for key, expected in q6.EXPECTED_READER.items():
            if int(reader.get(key, -1)) != int(expected):
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")
        expected_read = q6.K3_STREAM_BYTES * len(ORDER)
        if int(reader.get("bytes_read", -1)) != expected_read:
            raise RuntimeError(
                f"reader bytes_read={reader.get('bytes_read')} expected={expected_read}")

        ref = _means(records, "reference")
        pool2 = _means(records, "pool2")
        total_speedup = ref["seconds_mean"] / pool2["seconds_mean"]
        q6_speedup = ref["q6_boundary_seconds_mean"] / pool2["q6_boundary_seconds_mean"]
        non_q6_ratio = ref["non_q6_seconds_mean"] / pool2["non_q6_seconds_mean"]

        payload = {
            "schema": "qwen38-q6-persistent-pool-repeat-abba-v1",
            "status": "PASS",
            "claim": "counterbalanced repeated exact Q6_K-only 2-worker persistent-pool ABBA measurement on current-best real GGUF",
            "model_sha256": gdn.SHA256,
            "prompt_token_count": len(q6.PROMPT_IDS),
            "order": list(ORDER),
            "records": records,
            "reference_mean": ref,
            "pool2_mean": pool2,
            "speedup_total_mean": total_speedup,
            "speedup_q6_boundary_mean": q6_speedup,
            "non_q6_reference_to_pool_ratio": non_q6_ratio,
            "pool2_positive_total_mean": pool2["seconds_mean"] < ref["seconds_mean"],
            "pool2_positive_q6_mean": pool2["q6_boundary_seconds_mean"] < ref["q6_boundary_seconds_mean"],
            "q6_expected_coverage": q6.EXPECTED_Q6,
            "native_f32": native_f32.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "measurement_only": True,
                "q6_only": True,
                "q8_change": False,
                "persistent_workers": True,
                "threads": POOL_THREADS,
                "counterbalanced_order": "ABBA",
                "row_partition_only": True,
                "within_output_reduction_order_change": False,
                "q6_many_kernel_change": False,
                "arithmetic_change": False,
            },
            "baseline_q6_pool_ab_run": 33851935544,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_Q6_PERSISTENT_POOL_REPEAT_ABBA_REAL_BITWISE_PASS")
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
