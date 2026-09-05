#!/usr/bin/env python3
"""Offline scheduler replay for a measured progressive K3 deadline timeline.

The replay is measurement-only. It does not change the runtime, K3 bytes,
ring residency, storage concurrency, arithmetic, or quant scheduling.

It replays the single-I/O frontier service times measured by
qwen38_k3_deadline_timeline_probe.py against the measured main-thread event
order. Tensor-view wait time is removed from the fixed CPU-work clock and then
reintroduced by the simulated I/O readiness schedule. This lets us compare the
current layer-monolithic worker with cross-layer frontier scheduling policies
before touching the real reader.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

K3_STREAM_BYTES = 21_127_430_144
RING_SLOTS = 2
PLANNED_BYTES = 672_899_072

CPU_TYPES = {"bind", "prefetch", "tensor_view", "release"}


@dataclass(frozen=True)
class Task:
    layer: int
    frontier: int
    stage: str
    service: float
    nbytes: int
    file_offset: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.frontier)


@dataclass
class CpuEvent:
    typ: str
    layer: int
    frontier: int | None
    stage: str | None
    gap_before: float
    overhead: float
    original_start: float
    original_end: float
    accepted: bool | None = None
    wait_seconds_observed: float = 0.0


class Replay:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        policy: str,
        guard_seconds: float = 0.0,
    ) -> None:
        if policy not in {"monolithic", "edf", "slack"}:
            raise ValueError(policy)
        self.payload = payload
        self.policy = policy
        self.guard = float(guard_seconds)
        self.tasks = _tasks(payload)
        self.by_layer: dict[int, list[Task]] = defaultdict(list)
        for task in self.tasks:
            self.by_layer[task.layer].append(task)
        for layer in self.by_layer:
            self.by_layer[layer].sort(key=lambda x: x.frontier)

        self.cpu_events = _cpu_events(payload)
        self.deadline_work = _deadline_work(self.cpu_events, self.by_layer)

        self.wall = 0.0
        self.cpu_work = 0.0
        self.io_task: Task | None = None
        self.io_remaining = 0.0
        self.io_lock_layer: int | None = None
        self.eligible: dict[int, int] = {}
        self.request_seq = 0
        self.next_frontier: dict[int, int] = defaultdict(int)
        self.completed: set[tuple[int, int]] = set()
        self.first_io_time: dict[int, float] = {}
        self.prefetch_time: dict[int, float] = {}
        self.bound_layer: int | None = None
        self.tensor_wait = 0.0
        self.release_wait = 0.0
        self.tensor_wait_by_stage: dict[str, float] = defaultdict(float)
        self.io_order: list[tuple[int, int]] = []
        self.cross_layer_switches = 0
        self.file_offset_jumps = 0
        self.last_io_task: Task | None = None

    def _eligible_candidates(self) -> list[Task]:
        out: list[Task] = []
        for layer in sorted(self.eligible, key=lambda x: self.eligible[x]):
            idx = self.next_frontier[layer]
            layer_tasks = self.by_layer.get(layer, [])
            if idx >= len(layer_tasks):
                continue
            out.append(layer_tasks[idx])
        return out

    def _deadline_wall(self, task: Task) -> float:
        work = self.deadline_work.get(task.key, float("inf"))
        if work == float("inf"):
            return float("inf")
        return self.wall + max(0.0, work - self.cpu_work)

    def _choose_task(self) -> Task | None:
        candidates = self._eligible_candidates()
        if not candidates:
            return None

        if self.policy == "monolithic":
            if self.io_lock_layer is not None:
                for task in candidates:
                    if task.layer == self.io_lock_layer:
                        return task
            chosen = min(candidates, key=lambda t: self.eligible[t.layer])
            self.io_lock_layer = chosen.layer
            return chosen

        if self.policy == "edf":
            return min(
                candidates,
                key=lambda t: (
                    self._deadline_wall(t),
                    0 if t.layer == self.bound_layer else 1,
                    self.eligible[t.layer],
                ),
            )

        # Conservative slack stealing:
        # - default to the currently bound layer;
        # - allow a prefetched layer frontier only when its full measured
        #   service fits inside the bound layer's next-frontier slack,
        #   minus a configurable guard.
        current = None
        if self.bound_layer is not None:
            for task in candidates:
                if task.layer == self.bound_layer:
                    current = task
                    break
        if current is None:
            return min(candidates, key=lambda t: self.eligible[t.layer])

        other = [t for t in candidates if t.layer != self.bound_layer]
        if not other:
            return current
        next_task = min(
            other,
            key=lambda t: (self._deadline_wall(t), self.eligible[t.layer]),
        )
        current_deadline = self._deadline_wall(current)
        current_slack = current_deadline - self.wall - current.service
        if current_slack >= next_task.service + self.guard:
            return next_task
        return current

    def _start_io_if_idle(self) -> None:
        if self.io_task is not None:
            return
        task = self._choose_task()
        if task is None:
            return
        self.io_task = task
        self.io_remaining = task.service
        if task.layer not in self.first_io_time:
            self.first_io_time[task.layer] = self.wall
        if self.last_io_task is not None:
            if self.last_io_task.layer != task.layer:
                self.cross_layer_switches += 1
            previous_end = (
                self.last_io_task.file_offset + self.last_io_task.nbytes
            )
            if previous_end != task.file_offset:
                self.file_offset_jumps += 1
        self.io_order.append(task.key)

    def _complete_io(self) -> None:
        assert self.io_task is not None
        task = self.io_task
        self.completed.add(task.key)
        self.next_frontier[task.layer] += 1
        self.last_io_task = task
        self.io_task = None
        self.io_remaining = 0.0
        if self.next_frontier[task.layer] >= len(self.by_layer[task.layer]):
            if self.io_lock_layer == task.layer:
                self.io_lock_layer = None
        self._start_io_if_idle()

    def _advance(self, delta: float, *, cpu_progress: bool) -> None:
        remain = max(0.0, float(delta))
        self._start_io_if_idle()
        while remain > 1e-15:
            if self.io_task is None:
                self.wall += remain
                if cpu_progress:
                    self.cpu_work += remain
                return
            step = min(remain, self.io_remaining)
            self.wall += step
            if cpu_progress:
                self.cpu_work += step
            self.io_remaining -= step
            remain -= step
            if self.io_remaining <= 1e-12:
                self._complete_io()

    def _wait_for(self, key: tuple[int, int], *, stage: str, release: bool) -> None:
        start = self.wall
        while key not in self.completed:
            self._start_io_if_idle()
            if self.io_task is None:
                raise RuntimeError(f"deadlock waiting for {key}")
            self._advance(self.io_remaining, cpu_progress=False)
        waited = self.wall - start
        if release:
            self.release_wait += waited
        else:
            self.tensor_wait += waited
            self.tensor_wait_by_stage[stage] += waited

    def _make_eligible(self, layer: int, *, prefetch: bool) -> None:
        if layer in self.eligible:
            return
        self.eligible[layer] = self.request_seq
        self.request_seq += 1
        if prefetch:
            self.prefetch_time[layer] = self.wall
        self._start_io_if_idle()

    def run(self) -> dict[str, Any]:
        if not self.cpu_events:
            raise RuntimeError("no CPU events")
        for event in self.cpu_events:
            self._advance(event.gap_before, cpu_progress=True)

            if event.typ == "bind":
                self._make_eligible(event.layer, prefetch=False)
                self.bound_layer = event.layer
            elif event.typ == "prefetch":
                if event.accepted is not False:
                    self._make_eligible(event.layer, prefetch=True)
            elif event.typ == "tensor_view":
                assert event.frontier is not None
                self._wait_for(
                    (event.layer, event.frontier),
                    stage=event.stage or "unknown",
                    release=False,
                )
            elif event.typ == "release":
                tasks = self.by_layer[event.layer]
                self._wait_for(
                    (event.layer, tasks[-1].frontier),
                    stage="release",
                    release=True,
                )
                if self.bound_layer == event.layer:
                    self.bound_layer = None

            self._advance(event.overhead, cpu_progress=True)

        first_io_delays = []
        for layer, ptime in sorted(self.prefetch_time.items()):
            if layer in self.first_io_time:
                first_io_delays.append(self.first_io_time[layer] - ptime)

        return {
            "policy": self.policy,
            "guard_seconds": self.guard,
            "wall_seconds": self.wall,
            "cpu_work_seconds": self.cpu_work,
            "tensor_wait_seconds": self.tensor_wait,
            "release_wait_seconds": self.release_wait,
            "total_wait_seconds": self.tensor_wait + self.release_wait,
            "tensor_wait_by_stage": dict(
                sorted(self.tensor_wait_by_stage.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "completed_frontiers": len(self.completed),
            "io_service_seconds_sum": sum(t.service for t in self.tasks),
            "cross_layer_switches": self.cross_layer_switches,
            "file_offset_jumps": self.file_offset_jumps,
            "prefetch_to_first_io": {
                "layers": len(first_io_delays),
                "mean_seconds": (
                    sum(first_io_delays) / len(first_io_delays)
                    if first_io_delays else None
                ),
                "max_seconds": max(first_io_delays) if first_io_delays else None,
            },
            "io_order_head": [
                {"layer": layer, "frontier": frontier}
                for layer, frontier in self.io_order[:32]
            ],
        }


def _tasks(payload: dict[str, Any]) -> list[Task]:
    reads = [e for e in payload["events"] if e.get("type") == "io_frontier"]
    tasks = []
    for e in reads:
        tasks.append(Task(
            layer=int(e["layer"]),
            frontier=int(e["frontier_index"]),
            stage=str(e["stage"]),
            service=float(e["duration_s"]),
            nbytes=int(e["bytes"]),
            file_offset=int(e["file_offset"]),
        ))
    tasks.sort(key=lambda t: (t.layer, t.frontier))
    return tasks


def _cpu_events(payload: dict[str, Any]) -> list[CpuEvent]:
    raw = [e for e in payload["events"] if e.get("type") in CPU_TYPES]
    raw.sort(key=lambda e: (float(e["start_s"]), str(e["type"])))
    if not raw:
        return []
    out: list[CpuEvent] = []
    prev_end = float(raw[0]["start_s"])
    for e in raw:
        start = float(e["start_s"])
        end = float(e["end_s"])
        wait = float(e.get("wait_seconds", 0.0)) if e["type"] == "tensor_view" else 0.0
        overhead = max(0.0, (end - start) - wait)
        gap = max(0.0, start - prev_end)
        out.append(CpuEvent(
            typ=str(e["type"]),
            layer=int(e["layer"]),
            frontier=(
                int(e["frontier_index"]) if e["type"] == "tensor_view" else None
            ),
            stage=(str(e.get("stage", "unknown")) if e["type"] == "tensor_view" else None),
            gap_before=gap,
            overhead=overhead,
            original_start=start,
            original_end=end,
            accepted=(bool(e.get("accepted")) if e["type"] == "prefetch" else None),
            wait_seconds_observed=wait,
        ))
        prev_end = max(prev_end, end)
    return out


def _deadline_work(
    events: list[CpuEvent],
    by_layer: dict[int, list[Task]],
) -> dict[tuple[int, int], float]:
    cpu_work = 0.0
    demand: dict[tuple[int, int], float] = {}
    release_work: dict[int, float] = {}
    for e in events:
        cpu_work += e.gap_before
        if e.typ == "tensor_view":
            assert e.frontier is not None
            demand.setdefault((e.layer, e.frontier), cpu_work)
        elif e.typ == "release":
            release_work[e.layer] = cpu_work
        cpu_work += e.overhead

    for layer, tasks in by_layer.items():
        if tasks and layer in release_work:
            key = tasks[-1].key
            demand[key] = min(demand.get(key, float("inf")), release_work[layer])
    return demand


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("status") != "PASS":
        raise RuntimeError("timeline payload is not PASS")
    contracts = payload.get("contracts", {})
    if int(contracts.get("k3_bytes", -1)) != K3_STREAM_BYTES:
        raise RuntimeError("unexpected K3 byte contract")
    if int(contracts.get("ring_slots", -1)) != RING_SLOTS:
        raise RuntimeError("unexpected ring-slot contract")
    if int(contracts.get("planned_bytes", -1)) != PLANNED_BYTES:
        raise RuntimeError("unexpected residency contract")
    if int(contracts.get("storage_io_concurrency", -1)) != 1:
        raise RuntimeError("unexpected storage concurrency contract")
    if bool(contracts.get("arithmetic_change")):
        raise RuntimeError("timeline claims arithmetic change")

    tasks = _tasks(payload)
    if len(tasks) != 512:
        raise RuntimeError(f"expected 512 frontier tasks, got {len(tasks)}")
    if sum(t.nbytes for t in tasks) != K3_STREAM_BYTES:
        raise RuntimeError("frontier bytes do not equal K3 stream")
    by_layer: dict[int, list[Task]] = defaultdict(list)
    for t in tasks:
        by_layer[t.layer].append(t)
    if sorted(by_layer) != list(range(64)):
        raise RuntimeError("expected layers 0..63")
    if any(len(v) != 8 for v in by_layer.values()):
        raise RuntimeError("expected exactly 8 unique frontiers per layer")


def replay(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_payload(payload)
    observed_wait = float(payload["summary"]["tensor_wait_seconds_observed"])
    observed_span_events = [
        e for e in payload["events"] if e.get("type") in CPU_TYPES
    ]
    observed_span = (
        max(float(e["end_s"]) for e in observed_span_events)
        - min(float(e["start_s"]) for e in observed_span_events)
    )

    monolithic = Replay(payload, policy="monolithic").run()
    edf = Replay(payload, policy="edf").run()
    slack_results = []
    for guard in (0.0, 0.01, 0.025, 0.05, 0.10):
        slack_results.append(
            Replay(payload, policy="slack", guard_seconds=guard).run()
        )

    wait_error = abs(monolithic["tensor_wait_seconds"] - observed_wait)
    span_error = abs(monolithic["wall_seconds"] - observed_span)
    baseline_fidelity = {
        "observed_tensor_wait_seconds": observed_wait,
        "replayed_tensor_wait_seconds": monolithic["tensor_wait_seconds"],
        "tensor_wait_abs_error_seconds": wait_error,
        "tensor_wait_rel_error": wait_error / max(observed_wait, 1e-9),
        "observed_cpu_event_span_seconds": observed_span,
        "replayed_cpu_event_span_seconds": monolithic["wall_seconds"],
        "span_abs_error_seconds": span_error,
        "span_rel_error": span_error / max(observed_span, 1e-9),
    }

    def compare(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "policy": candidate["policy"],
            "guard_seconds": candidate["guard_seconds"],
            "wall_seconds": candidate["wall_seconds"],
            "wall_seconds_saved_vs_monolithic": (
                monolithic["wall_seconds"] - candidate["wall_seconds"]
            ),
            "speedup_vs_monolithic": (
                monolithic["wall_seconds"] / candidate["wall_seconds"]
                if candidate["wall_seconds"] > 0 else None
            ),
            "tensor_wait_seconds": candidate["tensor_wait_seconds"],
            "tensor_wait_seconds_saved_vs_monolithic": (
                monolithic["tensor_wait_seconds"] - candidate["tensor_wait_seconds"]
            ),
            "release_wait_seconds": candidate["release_wait_seconds"],
            "cross_layer_switches": candidate["cross_layer_switches"],
            "file_offset_jumps": candidate["file_offset_jumps"],
            "prefetch_to_first_io": candidate["prefetch_to_first_io"],
            "tensor_wait_by_stage": candidate["tensor_wait_by_stage"],
        }

    candidates = [edf, *slack_results]
    comparisons = [compare(x) for x in candidates]
    best = min(candidates, key=lambda x: x["wall_seconds"])

    # This is an analysis gate, not a performance PASS gate. Refuse to make
    # scheduling recommendations if the replay cannot reproduce the measured
    # monolithic wait closely enough.
    if baseline_fidelity["tensor_wait_rel_error"] > 0.05:
        status = "INVALID_BASELINE_REPLAY"
        recommendation = "do_not_implement_scheduler"
    else:
        status = "PASS"
        saved = monolithic["wall_seconds"] - best["wall_seconds"]
        recommendation = (
            "scheduler_ab_warranted"
            if saved >= 0.25 and best["release_wait_seconds"] <= 0.05
            else "scheduler_ab_not_warranted"
        )

    return {
        "schema": "qwen38-k3-deadline-replay-v1",
        "status": status,
        "measurement_only": True,
        "source_timeline": {
            "schema": payload.get("schema"),
            "event_count": int(payload["summary"]["event_count"]),
            "io_frontier_count": int(payload["summary"]["io_frontier_count"]),
            "k3_bytes": int(payload["contracts"]["k3_bytes"]),
        },
        "contracts": {
            "ring_slots": RING_SLOTS,
            "planned_bytes": PLANNED_BYTES,
            "storage_io_concurrency": 1,
            "scheduler_runtime_changed": False,
            "arithmetic_changed": False,
            "service_time_model": (
                "measured non-preemptive frontier durations; optimistic across "
                "cross-layer file-offset switches because storage service is held fixed"
            ),
        },
        "baseline_fidelity": baseline_fidelity,
        "monolithic": monolithic,
        "candidates": comparisons,
        "best_candidate": compare(best),
        "recommendation": recommendation,
    }


def synthetic_sanity() -> None:
    # Two layers, two frontiers each. Layer 1 is prefetched while layer 0 is
    # active. Layer 0 has CPU slack before its second frontier is demanded, so
    # EDF/slack can legally pull layer-1 frontier 0 forward.
    events: list[dict[str, Any]] = []
    def add(typ, start, end, layer, **kw):
        events.append({
            "type": typ, "start_s": start, "end_s": end,
            "duration_s": end-start, "layer": layer, **kw,
        })

    # I/O timings are only service-time samples; replay reschedules them.
    add("io_frontier", 0.00, 0.10, 0, frontier_index=0, stage="a",
        bytes=4096, file_offset=0)
    add("io_frontier", 0.10, 0.30, 0, frontier_index=1, stage="b",
        bytes=4096, file_offset=4096)
    add("io_frontier", 0.30, 0.40, 1, frontier_index=0, stage="a",
        bytes=4096, file_offset=8192)
    add("io_frontier", 0.40, 0.60, 1, frontier_index=1, stage="b",
        bytes=4096, file_offset=12288)

    add("bind", 0.00, 0.001, 0)
    add("prefetch", 0.002, 0.003, 1, accepted=True)
    add("tensor_view", 0.11, 0.111, 0, frontier_index=0, stage="a",
        wait_seconds=0.0)
    # 0.25 s CPU work before layer0 frontier1 demand.
    add("tensor_view", 0.361, 0.362, 0, frontier_index=1, stage="b",
        wait_seconds=0.0)
    add("release", 0.363, 0.364, 0)
    add("bind", 0.365, 0.366, 1)
    add("tensor_view", 0.367, 0.467, 1, frontier_index=0, stage="a",
        wait_seconds=0.099)
    add("tensor_view", 0.468, 0.668, 1, frontier_index=1, stage="b",
        wait_seconds=0.199)
    add("release", 0.669, 0.670, 1)

    payload = {
        "schema": "synthetic",
        "status": "PASS",
        "contracts": {
            "k3_bytes": 16384,
            "ring_slots": 2,
            "planned_bytes": PLANNED_BYTES,
            "storage_io_concurrency": 1,
            "arithmetic_change": False,
        },
        "summary": {
            "event_count": len(events),
            "io_frontier_count": 4,
            "tensor_wait_seconds_observed": 0.298,
        },
        "events": events,
    }
    # Test Replay directly because production contract validator expects 64x8.
    mono = Replay(payload, policy="monolithic").run()
    edf = Replay(payload, policy="edf").run()
    if len(mono["io_order_head"]) != 4 or len(edf["io_order_head"]) != 4:
        raise RuntimeError("synthetic replay lost I/O tasks")
    if edf["wall_seconds"] > mono["wall_seconds"] + 1e-9:
        raise RuntimeError("synthetic EDF unexpectedly regressed")
    print("QWEN38_K3_DEADLINE_REPLAY_SANITY PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--timeline", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "sanity":
        synthetic_sanity()
        return 0

    payload = json.loads(args.timeline.read_text(encoding="utf-8"))
    result = replay(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise RuntimeError(
            "deadline replay baseline did not reproduce the measured timeline"
        )
    print("QWEN38_K3_DEADLINE_REPLAY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
