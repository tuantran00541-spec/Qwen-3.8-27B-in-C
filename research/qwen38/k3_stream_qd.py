#!/usr/bin/env python3
"""Experimental bounded K3 reader with deeper lookahead and I/O queue depth.

This module is intentionally opt-in. The proven K3Trunk remains the default
runtime path. K3QDTrunk preserves the same bind/prefetch/tensor_view API while
allowing a bounded number of ring slots, prefetch lookahead layers, and
synchronous pread workers so we can measure whether NVMe queue depth improves
real direct-I/O service rate before committing extra RAM to the production
reader.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import mmap
import os
import threading
import time
from pathlib import Path

from k3_stream import K3Trunk as BaseK3Trunk, MemoryPlan


class K3QDTrunk(BaseK3Trunk):
    def __init__(
        self,
        bin_path: Path,
        index_path: Path,
        *,
        budget_bytes: int,
        want_ring: int = 2,
        max_pinned: int | None = None,
        prefer_direct_io: bool = True,
        ring_slots: int = 2,
        io_workers: int = 1,
        lookahead: int = 1,
    ) -> None:
        ring_slots = int(ring_slots)
        io_workers = int(io_workers)
        lookahead = int(lookahead)
        if ring_slots < 2 or ring_slots > 4:
            raise ValueError("experimental ring_slots must be in [2, 4]")
        if io_workers < 1 or io_workers > ring_slots - 1:
            raise ValueError("io_workers must be in [1, ring_slots-1]")
        if lookahead < 1 or lookahead > ring_slots - 1:
            raise ValueError("lookahead must be in [1, ring_slots-1]")
        if max_pinned not in (None, 0):
            raise ValueError("K3QDTrunk experiment does not combine lookahead with pinned layers")

        # Let the proven implementation validate the manifest, fd, direct-I/O
        # path and two-slot budget first. No layer I/O has happened yet.
        super().__init__(
            bin_path,
            index_path,
            budget_bytes=budget_bytes,
            want_ring=min(2, int(want_ring)),
            max_pinned=0,
            prefer_direct_io=prefer_direct_io,
        )

        slot_bytes = max(int(self.layers[layer]["read_bytes"]) for layer in self.order)
        planned = ring_slots * slot_bytes
        if planned > int(budget_bytes):
            super().close()
            raise MemoryError(
                f"experimental ring requires {planned} bytes, budget is {int(budget_bytes)}"
            )

        if self._executor is not None:
            self._executor.shutdown(wait=True)
        for buf in self._ring:
            buf.close()

        self.plan = MemoryPlan((), ring_slots, slot_bytes, planned, int(budget_bytes))
        self._ring = [mmap.mmap(-1, slot_bytes) for _ in range(ring_slots)]
        self._layer_of = [-1] * ring_slots
        self._slot_of = {}
        self._ring_cursor = 0
        self._active_slot = None
        self._executor = ThreadPoolExecutor(max_workers=io_workers)

        self._pending: dict[int, tuple[int, Future[int]]] = {}
        self._pending_by_slot: dict[int, int] = {}
        self._pending_order: deque[int] = deque()
        self._qd_io_workers = io_workers
        self._qd_lookahead = lookahead
        self._stats_lock = threading.Lock()
        self._qd_io_service_seconds = 0.0
        self._qd_io_bytes = 0
        self._qd_io_first_start: float | None = None
        self._qd_io_last_end: float | None = None
        self._qd_bind_wait_seconds = 0.0
        self._qd_bind_wait_calls = 0
        self._qd_prefetch_issued = 0
        self._qd_prefetch_ready_before_bind = 0
        self._qd_max_pending = 0

    def _load_layer(self, layer: int, target: mmap.mmap) -> int:
        started = time.perf_counter()
        got = super()._load_layer(layer, target)
        ended = time.perf_counter()
        elapsed = ended - started
        with self._stats_lock:
            self._qd_io_service_seconds += elapsed
            self._qd_io_bytes += int(got)
            if self._qd_io_first_start is None or started < self._qd_io_first_start:
                self._qd_io_first_start = started
            if self._qd_io_last_end is None or ended > self._qd_io_last_end:
                self._qd_io_last_end = ended
        return got

    def _finalize_pending(self, layer: int, *, count_wait: bool) -> bool:
        pending = self._pending.get(int(layer))
        if pending is None:
            return False
        slot, future = pending
        ready = future.done()
        t = time.perf_counter()
        future.result()
        waited = time.perf_counter() - t
        self._pending.pop(int(layer), None)
        self._pending_by_slot.pop(slot, None)
        try:
            self._pending_order.remove(int(layer))
        except ValueError:
            pass
        self._layer_of[slot] = int(layer)
        self._slot_of[int(layer)] = slot
        if ready:
            self._qd_prefetch_ready_before_bind += 1
        elif count_wait:
            self._qd_bind_wait_seconds += waited
            self._qd_bind_wait_calls += 1
        return True

    def _reap_done(self) -> None:
        for layer in list(self._pending_order):
            pending = self._pending.get(layer)
            if pending is not None and pending[1].done():
                self._finalize_pending(layer, count_wait=False)

    def _reserve_slot(self, anchor_layer: int) -> int | None:
        self._reap_done()
        n = len(self._ring)
        # Prefer empty/old slots. Never overwrite the active compute buffer,
        # a pending I/O target, or a future layer already prefetched.
        for pass_old in (True, False):
            for delta in range(n):
                slot = (self._ring_cursor + delta) % n
                if slot == self._active_slot or slot in self._pending_by_slot:
                    continue
                old = self._layer_of[slot]
                if pass_old:
                    if old >= int(anchor_layer):
                        continue
                elif old >= 0:
                    continue
                if old >= 0:
                    self._slot_of.pop(old, None)
                self._layer_of[slot] = -1
                self._ring_cursor = (slot + 1) % n
                return slot
        return None

    def _issue_one(self, layer: int, anchor_layer: int) -> bool:
        layer = int(layer)
        if layer not in self.layers or layer in self.plan.pinned_layers:
            return False
        if layer in self._slot_of or layer in self._pending:
            return False
        slot = self._reserve_slot(anchor_layer)
        if slot is None:
            return False
        future = self._executor.submit(self._load_layer, layer, self._ring[slot])
        self._pending[layer] = (slot, future)
        self._pending_by_slot[slot] = layer
        self._pending_order.append(layer)
        self._qd_prefetch_issued += 1
        self._qd_max_pending = max(self._qd_max_pending, len(self._pending))
        return True

    def bind(self, layer: int) -> memoryview:
        layer = int(layer)
        if layer not in self.layers:
            raise KeyError(layer)
        read_bytes = int(self.layers[layer]["read_bytes"])

        if self._finalize_pending(layer, count_wait=True):
            slot = self._slot_of[layer]
            self.hits += 1
            self._active_slot = slot
            return memoryview(self._ring[slot])[:read_bytes]

        slot = self._slot_of.get(layer)
        if slot is not None and self._layer_of[slot] == layer:
            self.hits += 1
            self._active_slot = slot
            return memoryview(self._ring[slot])[:read_bytes]

        # bind() is called only after the previous layer view was released, so
        # the previous active slot is reusable for a synchronous miss.
        self._active_slot = None
        slot = self._reserve_slot(layer + 1)
        if slot is None:
            # An unexpected miss can occur only if all slots contain useful
            # future work. Drain the oldest request, then evict the oldest
            # completed future rather than overwriting an in-flight buffer.
            if self._pending_order:
                self._finalize_pending(self._pending_order[0], count_wait=False)
            candidates = [
                i for i in range(len(self._ring))
                if i not in self._pending_by_slot
            ]
            if not candidates:
                raise RuntimeError("no safe K3 ring slot available for bind")
            slot = min(
                candidates,
                key=lambda i: self._layer_of[i] if self._layer_of[i] >= 0 else -1,
            )
            old = self._layer_of[slot]
            if old >= 0:
                self._slot_of.pop(old, None)
            self._layer_of[slot] = -1

        self._load_layer(layer, self._ring[slot])
        self._layer_of[slot] = layer
        self._slot_of[layer] = slot
        self.misses += 1
        self._active_slot = slot
        return memoryview(self._ring[slot])[:read_bytes]

    def prefetch(self, layer: int) -> bool:
        layer = int(layer)
        issued_requested = False
        last = max(self.order) + 1
        for target in range(layer, min(layer + self._qd_lookahead, last)):
            issued = self._issue_one(target, layer)
            if target == layer:
                issued_requested = issued
        return issued_requested

    def report(self) -> dict:
        with self._stats_lock:
            first = self._qd_io_first_start
            last = self._qd_io_last_end
            service = self._qd_io_service_seconds
            io_bytes = self._qd_io_bytes
        io_span = 0.0 if first is None or last is None else max(0.0, last - first)
        report = super().report()
        report.update({
            "experimental_qd": True,
            "io_workers": self._qd_io_workers,
            "lookahead_layers": self._qd_lookahead,
            "io_service_seconds_sum_nonadditive": service,
            "io_span_seconds": io_span,
            "io_bytes_completed": io_bytes,
            "bind_wait_seconds": self._qd_bind_wait_seconds,
            "bind_wait_calls": self._qd_bind_wait_calls,
            "prefetch_issued": self._qd_prefetch_issued,
            "prefetch_ready_before_bind": self._qd_prefetch_ready_before_bind,
            "max_pending": self._qd_max_pending,
        })
        return report

    def close(self) -> None:
        for layer in list(self._pending_order):
            pending = self._pending.get(layer)
            if pending is not None:
                pending[1].result()
        self._pending.clear()
        self._pending_by_slot.clear()
        self._pending_order.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        try:
            os.close(self.fd)
        finally:
            self.fd = -1
        for buf in list(self._pinned.values()) + self._ring:
            try:
                buf.close()
            except BufferError:
                pass
