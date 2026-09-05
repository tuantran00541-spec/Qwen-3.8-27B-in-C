#!/usr/bin/env python3
"""Win32 native-direct backend for the exact progressive K3 reader.

This module deliberately leaves the proven Linux ProgressiveK3Trunk untouched.
It reuses its queue/frontier/bind/prefetch/tensor-view/release lifecycle while
replacing only storage access and ring allocation:

- CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
- sector-aware alignment from FILE_STORAGE_INFO
- two VirtualAlloc-backed native ring buffers
- exactly one active ReadFile operation at a time

No tensor ordering, readiness frontier, scheduler, ring-count, or arithmetic
policy is changed.
"""
from __future__ import annotations

from collections import deque
import ctypes
import gc
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from k3_stream import ALIGN, _build_tensor_index, _layer_map, plan_memory
from qwen38_k3_progressive import (
    LAYOUT_POLICY,
    ProgressiveK3Trunk,
    _SlotState,
)
from qwen38_win32_direct_io import Win32DirectIO


class Win32ProgressiveK3Trunk(ProgressiveK3Trunk):
    """Progressive K3 lifecycle backed by native Win32 unbuffered reads."""

    def __init__(
        self,
        bin_path: Path,
        index_path: Path,
        *,
        budget_bytes: int,
        win32_direct_lib: Path,
        want_ring: int = 2,
        max_pinned: int | None = None,
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("Win32ProgressiveK3Trunk requires Windows")

        self.bin_path = Path(bin_path)
        self.manifest = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if self.manifest.get("layout_policy") != LAYOUT_POLICY:
            raise ValueError("Win32 progressive reader requires execution-order K3 manifest")
        if not bool(self.manifest.get("progressive_readiness")):
            raise ValueError("manifest does not declare progressive readiness")

        self.layers = _layer_map(self.manifest)
        self.tensor_index = _build_tensor_index(self.manifest, self.layers)
        self.order = sorted(self.layers)
        self.plan = plan_memory(
            self.manifest,
            int(budget_bytes),
            want_ring=want_ring,
            max_pinned=max_pinned,
        )
        if self.plan.pinned_layers:
            raise ValueError("Win32 progressive reader requires max_pinned=0")
        if self.plan.ring_slots != 2:
            raise ValueError("Win32 progressive reader requires exactly two ring slots")

        size = self.bin_path.stat().st_size
        for layer, meta in self.layers.items():
            if int(meta["file_offset"]) + int(meta["read_bytes"]) > size:
                raise ValueError(f"layer {layer}: trunk range exceeds file size")
            self._validated_frontiers(meta)

        self._win32 = Win32DirectIO(Path(win32_direct_lib))
        self._win32_file = self._win32.open(self.bin_path)
        self._win32_alignment = self._win32.alignment(self._win32_file)
        logical = int(self._win32_alignment["logical_sector"])
        physical = int(self._win32_alignment["physical_sector"])
        alignment = int(self._win32_alignment["alignment"])
        if logical <= 0 or physical <= 0 or alignment <= 0:
            self._win32.close(self._win32_file)
            raise RuntimeError(f"invalid Win32 storage alignment: {self._win32_alignment}")
        if ALIGN % logical or ALIGN % alignment:
            self._win32.close(self._win32_file)
            raise RuntimeError(
                "current 4096-byte K3 frontier alignment is incompatible with "
                f"Win32 storage alignment {self._win32_alignment}")
        if self.plan.slot_bytes % logical:
            self._win32.close(self._win32_file)
            raise RuntimeError("K3 slot size is not logical-sector aligned")

        self.direct_io = True
        self._native_buffers: list[int] = []
        self._ring_arrays: list[Any] = []
        self._ring: list[memoryview] = []
        self._native_by_ring_id: dict[int, int] = {}
        try:
            for _ in range(2):
                native = self._win32.create_buffer(self.plan.slot_bytes, alignment)
                ptr = self._win32.buffer_ptr(native)
                array_type = ctypes.c_ubyte * self.plan.slot_bytes
                array = array_type.from_address(ptr)
                ring = memoryview(array).cast("B")
                self._native_buffers.append(native)
                self._ring_arrays.append(array)
                self._ring.append(ring)
                self._native_by_ring_id[id(ring)] = native
        except BaseException:
            for ring in self._ring:
                try:
                    ring.release()
                except Exception:
                    pass
            for native in reversed(self._native_buffers):
                try:
                    self._win32.destroy_buffer(native)
                except Exception:
                    pass
            self._win32.close(self._win32_file)
            raise

        self._states = [_SlotState(), _SlotState()]
        self._layer_to_slot: dict[int, int] = {}
        self._active_slot: int | None = None
        self._cv = threading.Condition()
        self._queue: deque[tuple[int, int]] = deque()
        self._stop = False
        self._current_io_slot: int | None = None
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="qwen-k3-progressive-win32",
            daemon=True,
        )

        self.bytes_read = 0
        self.read_calls = 0
        self.ready_events = 0
        self.hits = 0
        self.misses = 0
        self.prefetch_issued = 0
        self.prefetch_rejected = 0
        self.tensor_wait_calls = 0
        self.tensor_wait_seconds = 0.0
        self.max_queued_requests_observed = 0
        self._closed = False
        self._worker.start()

    def _pread_range_into(
        self,
        target: memoryview,
        start: int,
        nbytes: int,
        file_offset: int,
    ) -> int:
        if start % ALIGN or nbytes % ALIGN or file_offset % ALIGN:
            raise ValueError("Win32 progressive direct-read ranges must be page aligned")
        native = self._native_by_ring_id.get(id(target))
        if native is None:
            raise RuntimeError("Win32 progressive read target is not a native ring slot")
        if nbytes > 0xFFFFFFFF:
            raise ValueError("single Win32 frontier exceeds ReadFile DWORD byte count")
        rc = self._win32.read(
            self._win32_file,
            native,
            buffer_offset=int(start),
            file_offset=int(file_offset),
            nbytes=int(nbytes),
        )
        if rc != 0:
            report = self._win32.report(self._win32_file)
            raise OSError(
                f"Win32 direct read failed rc={rc} start={start} bytes={nbytes} "
                f"file_offset={file_offset} report={report}")
        self.read_calls += 1
        self.bytes_read += int(nbytes)
        return int(nbytes)

    def report(self) -> dict[str, Any]:
        out = super().report()
        native = self._win32.report(self._win32_file)
        out.update({
            "platform": "win32",
            "direct_io": True,
            "win32_no_buffering": bool(native["no_buffering"]),
            "win32_overlapped": bool(native["overlapped"]),
            "win32_logical_sector": int(native["logical_sector"]),
            "win32_physical_sector": int(native["physical_sector"]),
            "win32_alignment": int(native["alignment"]),
            "win32_native_read_calls": int(native["calls"]),
            "win32_native_read_bytes": int(native["bytes"]),
            "win32_native_read_seconds": float(native["seconds"]),
            "win32_native_last_error": int(native["last_error"]),
            "native_ring_slots": len(self._native_buffers),
        })
        return out

    def close(self) -> None:
        if self._closed:
            return
        with self._cv:
            while self._current_io_slot is not None or self._queue:
                self._cv.wait()
            self._stop = True
            self._cv.notify_all()
        self._worker.join()

        # tensor_view slices are expected to be dead once the runtime closes.
        # Collect first, then release base exports before VirtualFree.
        gc.collect()
        for ring in self._ring:
            try:
                ring.release()
            except Exception:
                pass
        self._ring.clear()
        self._native_by_ring_id.clear()
        self._ring_arrays.clear()
        for native in reversed(self._native_buffers):
            self._win32.destroy_buffer(native)
        self._native_buffers.clear()
        self._win32.close(self._win32_file)
        self._closed = True


def _synthetic_manifest(root: Path) -> tuple[Path, Path, dict[str, bytes]]:
    trunk = root / "win32-progressive-fixture.bin"
    index = root / "win32-progressive-fixture.json"
    layer_bytes = 8 * ALIGN
    full = bytearray(2 * layer_bytes)
    expected: dict[str, bytes] = {}
    layers = []
    tensor_index = {}

    for layer in range(2):
        tensors = []
        frontiers = []
        base = layer * layer_bytes
        for frontier in range(8):
            name = f"blk.{layer}.fixture_{frontier}.weight"
            offset = frontier * ALIGN
            size = 173 + frontier * 17
            value = ((layer + 1) * 37 + frontier * 19) & 0xFF
            payload = bytes([value]) * size
            full[base + offset:base + offset + size] = payload
            expected[name] = payload
            tensors.append({
                "name": name,
                "layer": layer,
                "offset": offset,
                "nbytes": size,
            })
            tensor_index[name] = {
                "layer": layer,
                "offset": offset,
                "nbytes": size,
            }
            frontiers.append({
                "stage": f"fixture_stage_{frontier}",
                "ready_bytes": (frontier + 1) * ALIGN,
                "ready_fraction": (frontier + 1) / 8.0,
            })
        layers.append({
            "layer": layer,
            "kind": "synthetic",
            "file_offset": base,
            "data_bytes": 7 * ALIGN + len(expected[f"blk.{layer}.fixture_7.weight"]),
            "read_bytes": layer_bytes,
            "tensor_count": 8,
            "tensors": tensors,
            "readiness_frontiers": frontiers,
        })

    trunk.write_bytes(full)
    manifest = {
        "schema": "qwen38-k3-trunk-v1",
        "alignment": ALIGN,
        "tensor_alignment": 64,
        "layout_policy": LAYOUT_POLICY,
        "progressive_readiness": True,
        "storage_io_concurrency": 1,
        "max_deferred_layer_requests": 1,
        "layers": layers,
        "tensor_index": tensor_index,
        "total_read_bytes": len(full),
        "packed_file_bytes": len(full),
    }
    index.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return trunk, index, expected


def synthetic_lifecycle(win32_direct_lib: Path) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("Win32 progressive synthetic gate requires Windows")
    import hashlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        trunk, index, expected = _synthetic_manifest(root)
        budget = 2 * 8 * ALIGN
        reader = Win32ProgressiveK3Trunk(
            trunk,
            index,
            budget_bytes=budget,
            win32_direct_lib=win32_direct_lib,
            want_ring=2,
            max_pinned=0,
        )
        try:
            bound0 = reader.bind(0)
            if not reader.prefetch(1):
                raise RuntimeError("Win32 progressive prefetch(1) was not accepted")

            observed_names: list[str] = []
            for layer, bound in ((0, bound0),):
                for frontier in range(8):
                    name = f"blk.{layer}.fixture_{frontier}.weight"
                    view = reader.tensor_view(bound, name)
                    got = hashlib.sha256(view).hexdigest()
                    want = hashlib.sha256(expected[name]).hexdigest()
                    if got != want:
                        raise RuntimeError(f"{name}: SHA mismatch {got} != {want}")
                    observed_names.append(name)
                    view.release()
            bound0.release()

            bound1 = reader.bind(1)
            for frontier in range(8):
                name = f"blk.1.fixture_{frontier}.weight"
                view = reader.tensor_view(bound1, name)
                got = hashlib.sha256(view).hexdigest()
                want = hashlib.sha256(expected[name]).hexdigest()
                if got != want:
                    raise RuntimeError(f"{name}: SHA mismatch {got} != {want}")
                observed_names.append(name)
                view.release()
            bound1.release()

            report = reader.report()
            expected_bytes = 2 * 8 * ALIGN
            if len(observed_names) != 16:
                raise RuntimeError("Win32 progressive gate did not validate all tensors")
            if report["bytes_read"] != expected_bytes:
                raise RuntimeError(f"reader bytes mismatch: {report}")
            if report["win32_native_read_bytes"] != expected_bytes:
                raise RuntimeError(f"native bytes mismatch: {report}")
            if report["pread_calls"] != 16 or report["ready_events"] != 16:
                raise RuntimeError(f"frontier counters mismatch: {report}")
            if report["win32_native_read_calls"] != 16:
                raise RuntimeError(f"native call count mismatch: {report}")
            if report["ring_slots"] != 2 or report["native_ring_slots"] != 2:
                raise RuntimeError(f"ring contract mismatch: {report}")
            if report["planned_bytes"] != budget or report["budget_bytes"] != budget:
                raise RuntimeError(f"residency contract mismatch: {report}")
            if report["storage_io_concurrency"] != 1:
                raise RuntimeError(f"I/O concurrency contract mismatch: {report}")
            if report["max_deferred_layer_requests"] != 1:
                raise RuntimeError(f"deferred request contract mismatch: {report}")
            if report["max_queued_requests_observed"] > 1:
                raise RuntimeError(f"queue-depth contract violated: {report}")
            if not report["direct_io"] or not report["win32_no_buffering"]:
                raise RuntimeError(f"Win32 direct/no-buffering contract missing: {report}")
            if not report["win32_overlapped"]:
                raise RuntimeError(f"Win32 overlapped contract missing: {report}")

            result = {
                "status": "PASS",
                "tensor_sha_checks": len(observed_names),
                "fixture_read_bytes": expected_bytes,
                "reader": report,
                "scheduler_change": False,
                "layout_change": False,
                "arithmetic_change": False,
            }
            print(json.dumps(result, indent=2))
            print("QWEN38_WIN32_PROGRESSIVE_K3_SYNTHETIC_PASS")
            print("QWEN38_WIN32_PROGRESSIVE_K3_LIFECYCLE_PASS")
            return result
        finally:
            reader.close()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--win32-direct-lib", type=Path, required=True)
    args = ap.parse_args()
    synthetic_lifecycle(args.win32_direct_lib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
