#!/usr/bin/env python3
"""Same-run warm ABBA gate: whole-layer K3 vs progressive execution-ordered K3."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics

import qwen38_current_best_q6_profile as old
import qwen38_progressive_current_best_profile as progressive

ORDER = ("old-a", "progressive-a", "progressive-b", "old-b")


def parser() -> argparse.ArgumentParser:
    ap = old.parser()
    run = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction)).choices["run"]
    run.add_argument("--old-work-dir", type=Path)
    run.add_argument("--progressive-work-dir", type=Path)
    return ap


def _args_for(args, mode: str, label: str):
    out = copy.copy(args)
    out.work_dir = args.old_work_dir if mode == "old" else args.progressive_work_dir
    out.output = args.output.parent / f"{label}.json"
    return out


def _run_one(args, mode: str, label: str):
    fn = old.run if mode == "old" else progressive.run
    payload = fn(_args_for(args, mode, label))
    reader = payload["reader"]
    if payload["status"] != "PASS":
        raise RuntimeError(f"{label}: profile status is not PASS")
    if payload["hidden_sha256"] != old.KNOWN_HIDDEN_SHA256:
        raise RuntimeError(f"{label}: hidden anchor changed")
    if payload["state_sha256"] != old.KNOWN_STATE_SHA256:
        raise RuntimeError(f"{label}: state anchor changed")
    if int(payload["k3_bytes"]) != old.K3_STREAM_BYTES:
        raise RuntimeError(f"{label}: K3 byte contract changed")
    if not bool(reader.get("direct_io")):
        raise RuntimeError(f"{label}: direct I/O is not active")
    if int(reader.get("ring_slots", -1)) != 2 or int(reader.get("planned_bytes", -1)) != 672_899_072:
        raise RuntimeError(f"{label}: ring residency changed: {reader}")
    if mode == "old":
        if bool(reader.get("progressive_readiness", False)):
            raise RuntimeError(f"{label}: old reader unexpectedly reports progressive readiness")
    else:
        if not bool(reader.get("progressive_readiness")):
            raise RuntimeError(f"{label}: progressive readiness is not active")
        if int(reader.get("storage_io_concurrency", -1)) != 1:
            raise RuntimeError(f"{label}: progressive I/O concurrency changed")
        if int(reader.get("max_queued_requests_observed", 99)) > 1:
            raise RuntimeError(f"{label}: progressive deferred queue exceeded one request")
    return payload


def _compact(label: str, payload: dict) -> dict:
    reader = payload["reader"]
    return {
        "label": label,
        "mode": "progressive" if label.startswith("progressive") else "old",
        "prefill_seconds": float(payload["prefill_seconds"]),
        "q6_boundary_seconds": float(payload["q6_boundary_seconds"]),
        "reader_bind_seconds": float(payload["reader_bind_seconds"]),
        "tensor_wait_seconds": float(reader.get("tensor_wait_seconds", 0.0)),
        "k3_bytes": int(payload["k3_bytes"]),
        "hidden_sha256": payload["hidden_sha256"],
        "state_sha256": payload["state_sha256"],
        "max_rss_gib": float(payload["max_rss_gib"]),
    }


def sanity() -> None:
    old.sanity()
    progressive.sanity()
    if ORDER != ("old-a", "progressive-a", "progressive-b", "old-b"):
        raise RuntimeError("ABBA order changed")
    print("QWEN38_K3_PROGRESSIVE_ABBA_SANITY PASS")


def run(args) -> dict:
    if args.old_work_dir is None or args.progressive_work_dir is None:
        raise RuntimeError("--old-work-dir and --progressive-work-dir are required")
    if args.old_work_dir.resolve() == args.progressive_work_dir.resolve():
        raise RuntimeError("old/progressive trunks must use distinct work directories")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # One exact full-stack warm pass per layout creates/reuses its trunk before
    # the measured ABBA sequence and prevents first-pass packing/storage state
    # from being attributed to either candidate.
    warm_old = _run_one(args, "old", "warm-old")
    warm_prog = _run_one(args, "progressive", "warm-progressive")

    records = []
    for label in ORDER:
        mode = "progressive" if label.startswith("progressive") else "old"
        records.append(_compact(label, _run_one(args, mode, label)))

    old_rows = [r for r in records if r["mode"] == "old"]
    prog_rows = [r for r in records if r["mode"] == "progressive"]
    old_mean = statistics.fmean(r["prefill_seconds"] for r in old_rows)
    prog_mean = statistics.fmean(r["prefill_seconds"] for r in prog_rows)
    speedup = old_mean / prog_mean
    seconds_saved = old_mean - prog_mean

    payload = {
        "schema": "qwen38-k3-progressive-warm-abba-v1",
        "status": "PASS",
        "order": list(ORDER),
        "warmup": {
            "old_prefill_seconds": float(warm_old["prefill_seconds"]),
            "progressive_prefill_seconds": float(warm_prog["prefill_seconds"]),
        },
        "records": records,
        "means": {
            "old_prefill_seconds": old_mean,
            "progressive_prefill_seconds": prog_mean,
            "old_q6_boundary_seconds": statistics.fmean(r["q6_boundary_seconds"] for r in old_rows),
            "progressive_q6_boundary_seconds": statistics.fmean(r["q6_boundary_seconds"] for r in prog_rows),
            "old_bind_seconds": statistics.fmean(r["reader_bind_seconds"] for r in old_rows),
            "progressive_bind_seconds": statistics.fmean(r["reader_bind_seconds"] for r in prog_rows),
            "progressive_tensor_wait_seconds": statistics.fmean(r["tensor_wait_seconds"] for r in prog_rows),
        },
        "comparison": {
            "progressive_speedup_vs_old": speedup,
            "seconds_saved_per_11_token_prefill": seconds_saved,
            "material_ge_2pct": speedup >= 1.02,
            "regression_ge_2pct": speedup <= (1.0 / 1.02),
        },
        "contracts": {
            "hidden_sha256": old.KNOWN_HIDDEN_SHA256,
            "state_sha256": old.KNOWN_STATE_SHA256,
            "k3_bytes_per_pass": old.K3_STREAM_BYTES,
            "ring_slots": 2,
            "planned_bytes": 672_899_072,
            "progressive_storage_io_concurrency": 1,
            "arithmetic_change": False,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("QWEN38_K3_PROGRESSIVE_ABBA_REAL_BITWISE_PASS")
    return payload


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
