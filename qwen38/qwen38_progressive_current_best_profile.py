#!/usr/bin/env python3
"""Promoted current-best exact Q6-pool2 + progressive K3 profile.

Decoder arithmetic and promoted quant/native helpers are inherited from
qwen38_current_best_q6_profile.  The K3 policy is installed through the named
current-best overlay proven by the same-run warm ABBA gate.  The whole-layer
profile remains untouched as an explicit research control.
"""
from __future__ import annotations

import json

import qwen38_current_best_q6_profile as base
from qwen38_current_best_k3_runtime import (
    CURRENT_BEST_K3_LAYOUT_POLICY,
    CURRENT_BEST_K3_MAX_DEFERRED_LAYER_REQUESTS,
    CURRENT_BEST_K3_PLANNED_BYTES,
    CURRENT_BEST_K3_RING_SLOTS,
    CURRENT_BEST_K3_STORAGE_IO_CONCURRENCY,
    K3_STREAM_BYTES,
    PROOF_PROGRESSIVE_ABBA_ARTIFACT_ID,
    PROOF_PROGRESSIVE_ABBA_ARTIFACT_SHA256,
    PROOF_PROGRESSIVE_ABBA_COMMIT,
    PROOF_PROGRESSIVE_ABBA_RUN,
    PROOF_PROGRESSIVE_ABBA_SECONDS_SAVED,
    PROOF_PROGRESSIVE_ABBA_SPEEDUP,
    Qwen38CurrentBestK3Overlay,
    sanity as k3_runtime_sanity,
)


def sanity() -> None:
    base.sanity()
    k3_runtime_sanity()
    print("QWEN38_PROGRESSIVE_CURRENT_BEST_PROFILE_SANITY PASS")
    print("QWEN38_PROMOTED_CURRENT_BEST_SANITY PASS")


def run(args):
    with Qwen38CurrentBestK3Overlay() as k3_policy:
        payload = base.run(args)
        k3_policy_report = k3_policy.report()

    reader = payload["reader"]
    if reader.get("layout_policy") != CURRENT_BEST_K3_LAYOUT_POLICY:
        raise RuntimeError(f"promoted K3 layout policy missing from reader: {reader}")
    if not bool(reader.get("progressive_readiness")):
        raise RuntimeError("promoted progressive readiness was not active")
    if int(reader.get("storage_io_concurrency", -1)) != CURRENT_BEST_K3_STORAGE_IO_CONCURRENCY:
        raise RuntimeError("promoted profile changed storage I/O concurrency")
    if int(reader.get("ring_slots", -1)) != CURRENT_BEST_K3_RING_SLOTS:
        raise RuntimeError("promoted profile changed ring slots")
    if int(reader.get("planned_bytes", -1)) != CURRENT_BEST_K3_PLANNED_BYTES:
        raise RuntimeError("promoted profile changed ring residency budget")
    if int(reader.get("bytes_read", -1)) != K3_STREAM_BYTES:
        raise RuntimeError("promoted profile changed K3 stream bytes")
    if int(reader.get("max_queued_requests_observed", 99)) > CURRENT_BEST_K3_MAX_DEFERRED_LAYER_REQUESTS:
        raise RuntimeError("promoted profile exceeded one deferred layer request")

    wait_seconds = float(reader.get("tensor_wait_seconds", 0.0))
    raw_residual = float(payload["residual_non_bind_orchestration_seconds"])
    non_io_residual = raw_residual - wait_seconds
    if non_io_residual < -0.15:
        raise RuntimeError(
            f"promoted profile accounting became negative after tensor waits: {non_io_residual}")

    payload["schema"] = "qwen38-promoted-current-best-q6-progressive-k3-profile-v1"
    payload["claim"] = (
        "promoted Linux experimental current-best exact Q6-pool2 11-token decoder-prefill "
        "with execution-ordered two-slot progressive K3 readiness and one storage I/O at a time")
    payload["progressive_tensor_wait_seconds"] = wait_seconds
    payload["residual_non_bind_orchestration_seconds_before_progressive_wait_split"] = raw_residual
    payload["residual_non_io_orchestration_seconds"] = non_io_residual
    payload["current_best_k3_policy"] = k3_policy_report
    payload["promotion_evidence"] = {
        "same_run_warm_abba_run": PROOF_PROGRESSIVE_ABBA_RUN,
        "commit": PROOF_PROGRESSIVE_ABBA_COMMIT,
        "artifact_id": PROOF_PROGRESSIVE_ABBA_ARTIFACT_ID,
        "artifact_sha256": PROOF_PROGRESSIVE_ABBA_ARTIFACT_SHA256,
        "progressive_speedup_vs_whole_layer": PROOF_PROGRESSIVE_ABBA_SPEEDUP,
        "seconds_saved_per_11_token_prefill": PROOF_PROGRESSIVE_ABBA_SECONDS_SAVED,
        "material_ge_2pct": False,
    }
    payload["optimization"].update({
        "promoted_current_best": True,
        "full_runtime_promoted_current_best": True,
        "reader_policy_change": True,
        "execution_ordered_k3_layout": True,
        "progressive_tensor_readiness": True,
        "storage_io_concurrency": CURRENT_BEST_K3_STORAGE_IO_CONCURRENCY,
        "max_deferred_layer_requests": CURRENT_BEST_K3_MAX_DEFERRED_LAYER_REQUESTS,
        "ring_change": False,
        "ring_slots": CURRENT_BEST_K3_RING_SLOTS,
        "planned_bytes": CURRENT_BEST_K3_PLANNED_BYTES,
        "tensor_byte_change": False,
        "arithmetic_change": False,
    })
    payload["proof_sub_layer_frontier_run"] = 33863640313
    payload["proof_sub_layer_frontier_commit"] = "3ee9c823fc091cb579d14cd3fc8698331b20cdcd"
    payload["proof_progressive_abba_run"] = PROOF_PROGRESSIVE_ABBA_RUN
    payload["proof_progressive_abba_commit"] = PROOF_PROGRESSIVE_ABBA_COMMIT
    payload["proof_progressive_abba_artifact_id"] = PROOF_PROGRESSIVE_ABBA_ARTIFACT_ID
    payload["proof_progressive_abba_artifact_sha256"] = PROOF_PROGRESSIVE_ABBA_ARTIFACT_SHA256

    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "prefill_seconds": payload["prefill_seconds"],
        "k3_bytes": payload["k3_bytes"],
        "hidden_sha256": payload["hidden_sha256"],
        "state_sha256": payload["state_sha256"],
        "reader_bind_seconds": payload["reader_bind_seconds"],
        "progressive_tensor_wait_seconds": wait_seconds,
        "q6_boundary_seconds": payload["q6_boundary_seconds"],
        "residual_non_io_orchestration_seconds": non_io_residual,
        "promotion_evidence": payload["promotion_evidence"],
        "reader": reader,
        "max_rss_gib": payload["max_rss_gib"],
    }, indent=2, ensure_ascii=False))
    print("QWEN38_PROGRESSIVE_CURRENT_BEST_REAL_BITWISE_PASS")
    print("QWEN38_PROMOTED_CURRENT_BEST_REAL_BITWISE_PASS")
    return payload


def main() -> int:
    args = base.parser().parse_args()
    if args.mode == "sanity":
        sanity()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
