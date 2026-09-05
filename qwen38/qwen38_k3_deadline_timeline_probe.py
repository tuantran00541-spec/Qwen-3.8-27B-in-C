#!/usr/bin/env python3
"""Measurement-only deadline timeline for the promoted progressive K3 runtime.

No reader policy, packing policy, ring residency, I/O concurrency, decoder
arithmetic, or quant scheduling is changed.  The probe timestamps the existing
frontier reads, tensor readiness waits, layer lifecycle, and quant matvec calls
so a later scheduler experiment can be based on observed deadlines rather than
assumptions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any

import qwen38_current_best_q6_profile as base
import qwen38_progressive_current_best_profile as promoted
from q6_persistent_pool_runtime import Q6PersistentPoolRuntime
from qwen38_k3_progressive import ProgressiveK3Trunk


class Recorder:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def now(self) -> float:
        return time.monotonic() - self.t0

    def add(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)


def _layer_from_name(name: str) -> int:
    try:
        if name.startswith("blk."):
            return int(name.split(".", 2)[1])
    except Exception:
        pass
    return -1


def _role(name: str) -> str:
    if name.startswith("blk."):
        parts = name.split(".", 2)
        if len(parts) == 3:
            return parts[2]
    return name


def _frontier_for_required(reader: ProgressiveK3Trunk, layer: int, required: int) -> tuple[int, str, int]:
    raw = reader.layers[layer]["readiness_frontiers"]
    seen: list[tuple[int, str]] = []
    for item in raw:
        ready = int(item["ready_bytes"])
        stage = str(item["stage"])
        if seen and ready == seen[-1][0]:
            seen[-1] = (ready, seen[-1][1] + "+" + stage)
        else:
            seen.append((ready, stage))
    for idx, (ready, stage) in enumerate(seen):
        if ready >= required:
            return idx, stage, ready
    raise RuntimeError(f"layer {layer}: no frontier covers required={required}")


class Instrumentation:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec
        self.originals: dict[tuple[type, str], Any] = {}

    def _patch(self, cls: type, name: str, fn) -> None:
        self.originals[(cls, name)] = getattr(cls, name)
        setattr(cls, name, fn)

    def install(self) -> None:
        rec = self.rec

        orig_read = ProgressiveK3Trunk._pread_range_into
        def timed_read(reader, target, start, nbytes, file_offset):
            with reader._cv:
                slot = reader._current_io_slot
                layer = -1 if slot is None else int(reader._states[slot].layer)
            frontier_idx = -1
            stage = "unknown"
            ready = int(start) + int(nbytes)
            if layer >= 0:
                vals = []
                for item in reader.layers[layer]["readiness_frontiers"]:
                    r = int(item["ready_bytes"])
                    s = str(item["stage"])
                    if vals and r == vals[-1][0]:
                        vals[-1] = (r, vals[-1][1] + "+" + s)
                    else:
                        vals.append((r, s))
                for idx, (r, s) in enumerate(vals):
                    if r == ready:
                        frontier_idx, stage = idx, s
                        break
            t0 = rec.now()
            try:
                return orig_read(reader, target, start, nbytes, file_offset)
            finally:
                t1 = rec.now()
                rec.add({
                    "type": "io_frontier",
                    "thread": threading.current_thread().name,
                    "layer": layer,
                    "frontier_index": frontier_idx,
                    "stage": stage,
                    "start_s": t0,
                    "end_s": t1,
                    "duration_s": t1 - t0,
                    "slot_offset": int(start),
                    "bytes": int(nbytes),
                    "ready_bytes": ready,
                    "file_offset": int(file_offset),
                })
        self._patch(ProgressiveK3Trunk, "_pread_range_into", timed_read)

        orig_bind = ProgressiveK3Trunk.bind
        def timed_bind(reader, layer):
            t0 = rec.now()
            out = orig_bind(reader, layer)
            t1 = rec.now()
            rec.add({"type": "bind", "layer": int(layer), "start_s": t0, "end_s": t1, "duration_s": t1-t0})
            return out
        self._patch(ProgressiveK3Trunk, "bind", timed_bind)

        orig_prefetch = ProgressiveK3Trunk.prefetch
        def timed_prefetch(reader, layer):
            t0 = rec.now()
            accepted = orig_prefetch(reader, layer)
            t1 = rec.now()
            rec.add({
                "type": "prefetch", "layer": int(layer), "start_s": t0, "end_s": t1,
                "duration_s": t1-t0, "accepted": bool(accepted),
            })
            return accepted
        self._patch(ProgressiveK3Trunk, "prefetch", timed_prefetch)

        orig_view = ProgressiveK3Trunk.tensor_view
        def timed_view(reader, bound, tensor_name):
            meta = reader.tensor_index[str(tensor_name)]
            start = int(meta["offset"])
            required = min(
                int(reader.layers[int(bound.layer)]["read_bytes"]),
                ((start + int(meta["nbytes"]) + 4095) // 4096) * 4096,
            )
            fidx, stage, frontier_ready = _frontier_for_required(
                reader, int(bound.layer), required)
            before_calls = int(reader.tensor_wait_calls)
            before_wait = float(reader.tensor_wait_seconds)
            t0 = rec.now()
            out = orig_view(reader, bound, tensor_name)
            t1 = rec.now()
            after_calls = int(reader.tensor_wait_calls)
            after_wait = float(reader.tensor_wait_seconds)
            rec.add({
                "type": "tensor_view",
                "layer": int(bound.layer),
                "tensor": str(tensor_name),
                "role": _role(str(tensor_name)),
                "frontier_index": fidx,
                "stage": stage,
                "required_bytes": required,
                "frontier_ready_bytes": frontier_ready,
                "start_s": t0,
                "end_s": t1,
                "duration_s": t1-t0,
                "waited": after_calls > before_calls,
                "wait_seconds": max(0.0, after_wait - before_wait),
            })
            return out
        self._patch(ProgressiveK3Trunk, "tensor_view", timed_view)

        orig_release = ProgressiveK3Trunk._release_bound
        def timed_release(reader, bound):
            t0 = rec.now()
            out = orig_release(reader, bound)
            t1 = rec.now()
            rec.add({
                "type": "release", "layer": int(bound.layer),
                "start_s": t0, "end_s": t1, "duration_s": t1-t0,
            })
            return out
        self._patch(ProgressiveK3Trunk, "_release_bound", timed_release)

        orig_many = Q6PersistentPoolRuntime.matvec_many
        def timed_many(runtime, weights, meta, xs, prepared=None):
            name = str(meta.get("name", ""))
            t0 = rec.now()
            try:
                return orig_many(runtime, weights, meta, xs, prepared=prepared)
            finally:
                t1 = rec.now()
                rec.add({
                    "type": "matvec",
                    "layer": _layer_from_name(name),
                    "tensor": name,
                    "role": _role(name),
                    "quant": str(meta.get("type_name", "unknown")),
                    "start_s": t0,
                    "end_s": t1,
                    "duration_s": t1-t0,
                    "vectors": len(xs),
                })
        self._patch(Q6PersistentPoolRuntime, "matvec_many", timed_many)

    def restore(self) -> None:
        for (cls, name), fn in reversed(list(self.originals.items())):
            setattr(cls, name, fn)
        self.originals.clear()


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    reads = [e for e in events if e["type"] == "io_frontier"]
    views = [e for e in events if e["type"] == "tensor_view"]
    prefetch = [e for e in events if e["type"] == "prefetch" and e.get("accepted")]
    matvecs = [e for e in events if e["type"] == "matvec"]

    read_by_key = {(int(e["layer"]), int(e["frontier_index"])): e for e in reads}
    demand_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for e in views:
        demand_by_key[(int(e["layer"]), int(e["frontier_index"]))].append(e)

    frontier_rows = []
    for key, read in sorted(read_by_key.items()):
        demands = demand_by_key.get(key, [])
        first_demand = min((float(x["start_s"]) for x in demands), default=None)
        wait_s = sum(float(x["wait_seconds"]) for x in demands)
        wait_calls = sum(1 for x in demands if x["waited"])
        finish_before_demand = (
            max(0.0, first_demand - float(read["end_s"]))
            if first_demand is not None else None
        )
        lateness = (
            max(0.0, float(read["end_s"]) - first_demand)
            if first_demand is not None else None
        )
        frontier_rows.append({
            "layer": key[0],
            "frontier_index": key[1],
            "stage": read["stage"],
            "bytes": int(read["bytes"]),
            "service_seconds": float(read["duration_s"]),
            "read_start_s": float(read["start_s"]),
            "read_end_s": float(read["end_s"]),
            "first_demand_s": first_demand,
            "finish_before_first_demand_seconds": finish_before_demand,
            "read_finish_lateness_vs_first_demand_seconds": lateness,
            "tensor_calls": len(demands),
            "tensor_wait_calls": wait_calls,
            "tensor_wait_seconds": wait_s,
        })

    stage: dict[str, dict[str, Any]] = {}
    for row in frontier_rows:
        s = stage.setdefault(row["stage"], {
            "frontiers": 0, "bytes": 0, "service_seconds": 0.0,
            "tensor_calls": 0, "tensor_wait_calls": 0, "tensor_wait_seconds": 0.0,
            "finish_before_demand_margins": [], "late_frontiers": 0,
        })
        s["frontiers"] += 1
        s["bytes"] += int(row["bytes"])
        s["service_seconds"] += float(row["service_seconds"])
        s["tensor_calls"] += int(row["tensor_calls"])
        s["tensor_wait_calls"] += int(row["tensor_wait_calls"])
        s["tensor_wait_seconds"] += float(row["tensor_wait_seconds"])
        margin = row["finish_before_first_demand_seconds"]
        if margin is not None:
            s["finish_before_demand_margins"].append(float(margin))
        if row["read_finish_lateness_vs_first_demand_seconds"] and row["read_finish_lateness_vs_first_demand_seconds"] > 0:
            s["late_frontiers"] += 1

    stage_summary = {}
    for name, s in stage.items():
        margins = s.pop("finish_before_demand_margins")
        s["mean_finish_before_first_demand_seconds"] = statistics.fmean(margins) if margins else None
        s["median_finish_before_first_demand_seconds"] = statistics.median(margins) if margins else None
        s["max_finish_before_first_demand_seconds"] = max(margins) if margins else None
        stage_summary[name] = s

    first_read_by_layer = {}
    for e in reads:
        layer = int(e["layer"])
        first_read_by_layer[layer] = min(
            float(e["start_s"]), first_read_by_layer.get(layer, float("inf")))
    queue_rows = []
    for e in prefetch:
        layer = int(e["layer"])
        if layer in first_read_by_layer:
            queue_rows.append({
                "layer": layer,
                "prefetch_s": float(e["end_s"]),
                "first_io_s": first_read_by_layer[layer],
                "prefetch_to_first_io_seconds": max(
                    0.0, first_read_by_layer[layer] - float(e["end_s"])),
            })

    wait_views = sorted(
        (e for e in views if e["waited"]),
        key=lambda e: float(e["wait_seconds"]),
        reverse=True,
    )
    margin_rows = sorted(
        (r for r in frontier_rows if r["finish_before_first_demand_seconds"] is not None),
        key=lambda r: float(r["finish_before_first_demand_seconds"]),
        reverse=True,
    )

    return {
        "event_count": len(events),
        "io_frontier_count": len(reads),
        "io_bytes": sum(int(e["bytes"]) for e in reads),
        "io_service_seconds_sum": sum(float(e["duration_s"]) for e in reads),
        "tensor_view_calls": len(views),
        "tensor_wait_calls_observed": sum(1 for e in views if e["waited"]),
        "tensor_wait_seconds_observed": sum(float(e["wait_seconds"]) for e in views),
        "matvec_calls": len(matvecs),
        "matvec_seconds_instrumented": sum(float(e["duration_s"]) for e in matvecs),
        "prefetch_to_first_io": {
            "layers": len(queue_rows),
            "mean_seconds": statistics.fmean(
                r["prefetch_to_first_io_seconds"] for r in queue_rows) if queue_rows else None,
            "median_seconds": statistics.median(
                r["prefetch_to_first_io_seconds"] for r in queue_rows) if queue_rows else None,
            "max_seconds": max(
                (r["prefetch_to_first_io_seconds"] for r in queue_rows), default=None),
            "rows": queue_rows,
        },
        "stage_summary": stage_summary,
        "top_tensor_waits": wait_views[:40],
        "top_finish_before_demand_margins": margin_rows[:40],
        "frontiers": frontier_rows,
    }


def parser() -> argparse.ArgumentParser:
    ap = base.parser()
    run = next(
        a for a in ap._actions if isinstance(a, argparse._SubParsersAction)
    ).choices["run"]
    run.add_argument("--timeline-output", type=Path, required=True)
    return ap


def sanity() -> None:
    promoted.sanity()
    print("QWEN38_K3_DEADLINE_TIMELINE_SANITY PASS")


def run(args) -> dict[str, Any]:
    rec = Recorder()
    inst = Instrumentation(rec)
    inst.install()
    try:
        promoted_payload = promoted.run(args)
    finally:
        inst.restore()

    events = sorted(rec.events, key=lambda e: (float(e.get("start_s", 0.0)), e["type"]))
    summary = _summarize(events)
    reader = promoted_payload["reader"]

    if promoted_payload["status"] != "PASS":
        raise RuntimeError("promoted profile did not PASS under timeline instrumentation")
    if promoted_payload["hidden_sha256"] != base.KNOWN_HIDDEN_SHA256:
        raise RuntimeError("timeline instrumentation changed hidden anchor")
    if promoted_payload["state_sha256"] != base.KNOWN_STATE_SHA256:
        raise RuntimeError("timeline instrumentation changed state anchor")
    if int(promoted_payload["k3_bytes"]) != base.K3_STREAM_BYTES:
        raise RuntimeError("timeline instrumentation changed K3 bytes")
    if int(reader.get("ring_slots", -1)) != 2:
        raise RuntimeError("timeline instrumentation changed ring slots")
    if int(reader.get("planned_bytes", -1)) != 672_899_072:
        raise RuntimeError("timeline instrumentation changed ring residency")
    if int(reader.get("storage_io_concurrency", -1)) != 1:
        raise RuntimeError("timeline instrumentation changed storage I/O concurrency")
    if int(summary["io_bytes"]) != base.K3_STREAM_BYTES:
        raise RuntimeError(
            f"timeline I/O bytes {summary['io_bytes']} != {base.K3_STREAM_BYTES}")
    if int(summary["io_frontier_count"]) != int(reader.get("pread_calls", -1)):
        raise RuntimeError(
            f"timeline frontier count {summary['io_frontier_count']} "
            f"!= reader pread calls {reader.get('pread_calls')}")
    if int(summary["tensor_wait_calls_observed"]) != int(reader.get("tensor_wait_calls", -1)):
        raise RuntimeError(
            f"timeline wait calls {summary['tensor_wait_calls_observed']} "
            f"!= reader wait calls {reader.get('tensor_wait_calls')}")
    wait_delta = abs(
        float(summary["tensor_wait_seconds_observed"])
        - float(reader.get("tensor_wait_seconds", 0.0)))
    if wait_delta > 0.02:
        raise RuntimeError(f"timeline wait accounting mismatch: delta={wait_delta}")

    payload = {
        "schema": "qwen38-k3-progressive-deadline-timeline-v1",
        "status": "PASS",
        "measurement_only": True,
        "claim": (
            "event timeline of the promoted exact progressive K3 path; "
            "no scheduler, ring, I/O-concurrency, tensor-byte, or arithmetic change"),
        "contracts": {
            "hidden_sha256": promoted_payload["hidden_sha256"],
            "state_sha256": promoted_payload["state_sha256"],
            "k3_bytes": int(promoted_payload["k3_bytes"]),
            "ring_slots": int(reader["ring_slots"]),
            "planned_bytes": int(reader["planned_bytes"]),
            "storage_io_concurrency": int(reader["storage_io_concurrency"]),
            "max_deferred_layer_requests": int(reader["max_deferred_layer_requests"]),
            "arithmetic_change": False,
            "scheduler_change": False,
        },
        "profile": {
            "prefill_seconds_instrumented": float(promoted_payload["prefill_seconds"]),
            "q6_boundary_seconds_instrumented": float(promoted_payload["q6_boundary_seconds"]),
            "progressive_tensor_wait_seconds": float(reader["tensor_wait_seconds"]),
            "reader_pread_calls": int(reader["pread_calls"]),
            "reader_ready_events": int(reader["ready_events"]),
            "max_rss_gib": float(promoted_payload["max_rss_gib"]),
        },
        "summary": summary,
        "events": events,
    }
    args.timeline_output.parent.mkdir(parents=True, exist_ok=True)
    args.timeline_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "contracts": payload["contracts"],
        "profile": payload["profile"],
        "event_count": summary["event_count"],
        "io_frontier_count": summary["io_frontier_count"],
        "tensor_wait_calls_observed": summary["tensor_wait_calls_observed"],
        "tensor_wait_seconds_observed": summary["tensor_wait_seconds_observed"],
        "prefetch_to_first_io": {
            k: v for k, v in summary["prefetch_to_first_io"].items() if k != "rows"
        },
        "stage_summary": summary["stage_summary"],
        "top_finish_before_demand_margins": summary["top_finish_before_demand_margins"][:12],
        "top_tensor_waits": summary["top_tensor_waits"][:12],
    }, indent=2, ensure_ascii=False))
    print("QWEN38_K3_DEADLINE_TIMELINE_REAL_BITWISE_PASS")
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
