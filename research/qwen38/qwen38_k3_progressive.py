#!/usr/bin/env python3
"""Execution-ordered K3 packing and progressive single-I/O reader.

This is an isolated research backend.  It preserves the two-slot K3 residency
budget and allows only one storage read operation at a time.  The only layout
change is tensor order inside each layer: tensors are packed in decoder
execution/dependency order instead of lexicographic name order.

A progressive layer is read as a sequence of page-aligned prefixes recorded in
the manifest.  ``tensor_view`` blocks only until the page containing the
requested tensor is complete, allowing compute to consume an earlier stable
prefix while the single background I/O worker fills later, disjoint pages of
the same ring slot.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Iterable

from gguf_k3_layout import (
    K3_SCHEMA,
    MANIFEST_VERSION,
    SOURCE_FORMAT,
    _copy_hash,
    _write_zeros,
    partition_tensors,
)
from gguf_stream import parse_gguf
from k3_stream import (
    ALIGN,
    TENSOR_ALIGN,
    _build_tensor_index,
    _layer_map,
    align_up,
    plan_memory,
)
from qwen38_k3_sub_layer_frontier_probe import (
    ATTN_ORDER,
    ATTN_STAGES,
    GDN_ORDER,
    GDN_STAGES,
)

N_LAYER = 64
K3_STREAM_BYTES = 21_127_430_144
MODEL_SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
LAYOUT_POLICY = "qwen38-execution-dependency-v1"


def _suffix(layer: int, name: str) -> str:
    prefix = f"blk.{layer}."
    if not name.startswith(prefix):
        raise ValueError(f"layer {layer}: unexpected tensor {name}")
    return name[len(prefix):]


def _order_contract(layer: int):
    if layer % 4 == 3:
        return ATTN_ORDER, ATTN_STAGES, "attention"
    return GDN_ORDER, GDN_STAGES, "gdn"


def _readiness_frontiers(
    ends: dict[str, int], stages, read_bytes: int
) -> list[dict[str, Any]]:
    frontier = 0
    out: list[dict[str, Any]] = []
    for stage_name, needed in stages:
        for suffix in needed:
            frontier = max(frontier, int(ends[suffix]))
        ready = min(int(read_bytes), align_up(frontier, ALIGN))
        if out and ready < int(out[-1]["ready_bytes"]):
            raise AssertionError("readiness frontier moved backwards")
        out.append({
            "stage": stage_name,
            "ready_bytes": ready,
            "ready_fraction": ready / float(read_bytes),
        })
    if not out or int(out[-1]["ready_bytes"]) != int(read_bytes):
        raise RuntimeError(
            f"final execution stage must cover the whole layer: final={out[-1] if out else None} read={read_bytes}")
    return out


def pack_gguf_layers_progressive(
    directory,
    out_bin: Path,
    out_index: Path,
    *,
    layers: Iterable[int] | None = None,
    model_id: str,
    revision: str,
    source_sha256: str,
    expected_layers: int = N_LAYER,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Pack GGUF decoder tensors in exact semantic execution order."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    grouped, auxiliary, globals_ = partition_tensors(directory, expected_layers=expected_layers)
    selected = list(range(expected_layers)) if layers is None else sorted({int(x) for x in layers})
    if not selected or any(x not in grouped for x in selected):
        raise ValueError("selected layer ids are invalid")

    out_bin = Path(out_bin)
    out_index = Path(out_index)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    manifest_layers: list[dict[str, Any]] = []
    total_tensor_bytes = 0
    max_copy_chunk = 0
    src_fd = os.open(directory.path, os.O_RDONLY)
    try:
        with out_bin.open("wb") as dst:
            for layer in selected:
                execution_order, stages, kind = _order_contract(layer)
                spans = {_suffix(layer, t.name): t for t in grouped[layer]}
                if set(spans) != set(execution_order):
                    raise RuntimeError(
                        f"layer {layer} ({kind}) tensor contract mismatch: "
                        f"missing={sorted(set(execution_order)-set(spans))} "
                        f"unexpected={sorted(set(spans)-set(execution_order))}"
                    )

                start = align_up(dst.tell(), ALIGN)
                _write_zeros(dst, start - dst.tell())
                tensors_meta: list[dict[str, Any]] = []
                ends: dict[str, int] = {}
                for suffix in execution_order:
                    tensor = spans[suffix]
                    rel = align_up(dst.tell() - start, TENSOR_ALIGN)
                    absolute = start + rel
                    _write_zeros(dst, absolute - dst.tell())
                    digest, observed = _copy_hash(
                        src_fd, tensor.data_offset, tensor.nbytes, dst, chunk_bytes)
                    max_copy_chunk = max(max_copy_chunk, observed)
                    ends[suffix] = rel + int(tensor.nbytes)
                    tensors_meta.append({
                        "name": tensor.name,
                        "layer": layer,
                        "ggml_type": tensor.ggml_type,
                        "type_name": tensor.type_name,
                        "shape": list(tensor.shape),
                        "source_offset": tensor.data_offset,
                        "nbytes": tensor.nbytes,
                        "offset": rel,
                        "sha256": digest,
                    })
                    total_tensor_bytes += int(tensor.nbytes)

                data_bytes = dst.tell() - start
                end = align_up(dst.tell(), ALIGN)
                _write_zeros(dst, end - dst.tell())
                read_bytes = end - start
                frontiers = _readiness_frontiers(ends, stages, read_bytes)
                manifest_layers.append({
                    "layer": layer,
                    "kind": kind,
                    "file_offset": start,
                    "data_bytes": data_bytes,
                    "read_bytes": read_bytes,
                    "tensor_count": len(tensors_meta),
                    "tensors": tensors_meta,
                    "readiness_frontiers": frontiers,
                })
    finally:
        os.close(src_fd)

    tensor_index = {
        tensor["name"]: {
            "layer": layer["layer"],
            "offset": tensor["offset"],
            "nbytes": tensor["nbytes"],
        }
        for layer in manifest_layers
        for tensor in layer["tensors"]
    }
    manifest = {
        "schema": K3_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "source_format": SOURCE_FORMAT,
        "model_id": model_id,
        "revision": revision,
        "source": {
            "path": directory.path.name,
            "sha256": source_sha256,
            "file_bytes": directory.file_bytes,
        },
        "alignment": ALIGN,
        "tensor_alignment": TENSOR_ALIGN,
        "layout_policy": LAYOUT_POLICY,
        "progressive_readiness": True,
        "storage_io_concurrency": 1,
        "max_deferred_layer_requests": 1,
        "layers": manifest_layers,
        "auxiliary_blocks": [{
            "block": block,
            "tensor_count": len(tensors),
            "tensors": [{
                "name": t.name,
                "ggml_type": t.ggml_type,
                "type_name": t.type_name,
                "shape": list(t.shape),
                "source_offset": t.data_offset,
                "nbytes": t.nbytes,
            } for t in tensors],
        } for block, tensors in sorted(auxiliary.items())],
        "globals": [{
            "name": t.name,
            "ggml_type": t.ggml_type,
            "type_name": t.type_name,
            "shape": list(t.shape),
            "source_offset": t.data_offset,
            "nbytes": t.nbytes,
        } for t in globals_],
        "tensor_index": tensor_index,
        "total_tensor_bytes": total_tensor_bytes,
        "packed_file_bytes": out_bin.stat().st_size,
        "total_read_bytes": sum(int(x["read_bytes"]) for x in manifest_layers),
        "copy_chunk_bytes": chunk_bytes,
        "max_copy_chunk_observed": max_copy_chunk,
    }
    if selected == list(range(N_LAYER)) and int(manifest["total_read_bytes"]) != K3_STREAM_BYTES:
        raise RuntimeError(
            f"execution layout changed K3 stream bytes: {manifest['total_read_bytes']} != {K3_STREAM_BYTES}")
    out_index.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


@dataclass
class _SlotState:
    layer: int = -1
    status: str = "free"  # free/queued/loading/complete/error
    ready_bytes: int = 0
    read_bytes: int = 0
    active: bool = False
    error: BaseException | None = None


class ProgressiveBoundLayer:
    def __init__(self, reader: "ProgressiveK3Trunk", layer: int, slot: int) -> None:
        self.reader = reader
        self.layer = int(layer)
        self.slot = int(slot)
        self.released = False

    def release(self) -> None:
        if not self.released:
            self.reader._release_bound(self)
            self.released = True


class ProgressiveK3Trunk:
    """Two-slot, one-I/O-at-a-time progressive K3 reader."""

    def __init__(
        self,
        bin_path: Path,
        index_path: Path,
        *,
        budget_bytes: int,
        want_ring: int = 2,
        max_pinned: int | None = None,
        prefer_direct_io: bool = True,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.manifest = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if self.manifest.get("layout_policy") != LAYOUT_POLICY:
            raise ValueError("progressive reader requires execution-order K3 manifest")
        if not bool(self.manifest.get("progressive_readiness")):
            raise ValueError("manifest does not declare progressive readiness")
        self.layers = _layer_map(self.manifest)
        self.tensor_index = _build_tensor_index(self.manifest, self.layers)
        self.order = sorted(self.layers)
        self.plan = plan_memory(
            self.manifest, int(budget_bytes), want_ring=want_ring, max_pinned=max_pinned)
        if self.plan.pinned_layers:
            raise ValueError("experimental progressive reader requires max_pinned=0")
        if self.plan.ring_slots != 2:
            raise ValueError("experimental progressive reader requires exactly two ring slots")

        self.direct_io = False
        self._fd_lock = threading.Lock()
        self.fd = self._open_fd(prefer_direct_io)
        size = self.bin_path.stat().st_size
        for layer, meta in self.layers.items():
            if int(meta["file_offset"]) + int(meta["read_bytes"]) > size:
                raise ValueError(f"layer {layer}: trunk range exceeds file size")
            self._validated_frontiers(meta)

        self._ring = [mmap.mmap(-1, self.plan.slot_bytes) for _ in range(2)]
        self._states = [_SlotState(), _SlotState()]
        self._layer_to_slot: dict[int, int] = {}
        self._active_slot: int | None = None
        self._cv = threading.Condition()
        self._queue: deque[tuple[int, int]] = deque()
        self._stop = False
        self._current_io_slot: int | None = None
        self._worker = threading.Thread(
            target=self._worker_loop, name="qwen-k3-progressive", daemon=True)

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
        self._worker.start()

    def _open_fd(self, prefer_direct: bool) -> int:
        if prefer_direct and hasattr(os, "O_DIRECT"):
            try:
                fd = os.open(self.bin_path, os.O_RDONLY | os.O_DIRECT)
                self.direct_io = True
                return fd
            except OSError:
                pass
        self.direct_io = False
        return os.open(self.bin_path, os.O_RDONLY)

    def _fallback_buffered(self) -> None:
        with self._fd_lock:
            if not self.direct_io:
                return
            os.close(self.fd)
            self.fd = os.open(self.bin_path, os.O_RDONLY)
            self.direct_io = False

    def _validated_frontiers(self, meta: dict[str, Any]) -> list[int]:
        read_bytes = int(meta["read_bytes"])
        raw = meta.get("readiness_frontiers")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"layer {meta.get('layer')}: missing readiness frontiers")
        vals: list[int] = []
        for item in raw:
            value = int(item["ready_bytes"])
            if value <= 0 or value > read_bytes or value % ALIGN:
                raise ValueError(f"layer {meta.get('layer')}: invalid readiness frontier {value}")
            if not vals or value != vals[-1]:
                if vals and value < vals[-1]:
                    raise ValueError("readiness frontiers are not monotonic")
                vals.append(value)
        if vals[-1] != read_bytes:
            raise ValueError("final readiness frontier must equal read_bytes")
        return vals

    def _pread_range_into(self, target: mmap.mmap, start: int, nbytes: int, file_offset: int) -> int:
        if start % ALIGN or nbytes % ALIGN or file_offset % ALIGN:
            raise ValueError("progressive direct-read ranges must be page aligned")
        view = memoryview(target)[start:start + nbytes]
        done = 0
        while done < nbytes:
            try:
                if hasattr(os, "preadv"):
                    got = os.preadv(self.fd, [view[done:]], file_offset + done)
                else:
                    chunk = os.pread(self.fd, nbytes - done, file_offset + done)
                    got = len(chunk)
                    view[done:done + got] = chunk
            except OSError as exc:
                if self.direct_io and exc.errno in (22, 95):
                    self._fallback_buffered()
                    continue
                view.release()
                raise
            self.read_calls += 1
            if got <= 0:
                view.release()
                raise IOError(f"short read at {file_offset + done}")
            done += got
        view.release()
        self.bytes_read += done
        return done

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop and not self._queue:
                    return
                layer, slot = self._queue.popleft()
                state = self._states[slot]
                if state.layer != layer or state.status != "queued":
                    state.error = RuntimeError("progressive queue state mismatch")
                    state.status = "error"
                    self._cv.notify_all()
                    continue
                state.status = "loading"
                self._current_io_slot = slot
                self._cv.notify_all()

            try:
                meta = self.layers[layer]
                base = int(meta["file_offset"])
                previous = 0
                for ready in self._validated_frontiers(meta):
                    amount = ready - previous
                    if amount:
                        self._pread_range_into(self._ring[slot], previous, amount, base + previous)
                    with self._cv:
                        state = self._states[slot]
                        if state.layer != layer or state.status != "loading":
                            raise RuntimeError("slot changed while progressive I/O was active")
                        state.ready_bytes = ready
                        self.ready_events += 1
                        self._cv.notify_all()
                    previous = ready
                with self._cv:
                    state = self._states[slot]
                    state.status = "complete"
                    self._current_io_slot = None
                    self._cv.notify_all()
            except BaseException as exc:
                with self._cv:
                    state = self._states[slot]
                    state.error = exc
                    state.status = "error"
                    self._current_io_slot = None
                    self._cv.notify_all()

    def _choose_slot_locked(self) -> int | None:
        for slot, state in enumerate(self._states):
            if state.active or state.status in {"queued", "loading"}:
                continue
            return slot
        return None

    def _assign_queued_locked(self, layer: int, slot: int) -> None:
        state = self._states[slot]
        if state.layer >= 0:
            self._layer_to_slot.pop(state.layer, None)
        self._states[slot] = _SlotState(
            layer=int(layer), status="queued", ready_bytes=0,
            read_bytes=int(self.layers[layer]["read_bytes"]), active=False, error=None)
        self._layer_to_slot[int(layer)] = slot
        self._queue.append((int(layer), slot))
        self.max_queued_requests_observed = max(
            self.max_queued_requests_observed, len(self._queue))
        if len(self._queue) > 1:
            raise RuntimeError("progressive reader exceeded one deferred layer request")
        self._cv.notify_all()

    def bind(self, layer: int) -> ProgressiveBoundLayer:
        layer = int(layer)
        if layer not in self.layers:
            raise KeyError(layer)
        with self._cv:
            slot = self._layer_to_slot.get(layer)
            if slot is None:
                slot = self._choose_slot_locked()
                if slot is None:
                    raise RuntimeError(f"no safe ring slot available to bind layer {layer}")
                self._assign_queued_locked(layer, slot)
                self.misses += 1
            else:
                self.hits += 1
            state = self._states[slot]
            while state.status == "queued":
                self._cv.wait()
                state = self._states[slot]
            if state.status == "error":
                assert state.error is not None
                raise RuntimeError(f"layer {layer} progressive I/O failed") from state.error
            if state.layer != layer or state.status not in {"loading", "complete"}:
                raise RuntimeError(f"layer {layer}: invalid bind state {state}")
            if self._active_slot is not None and self._active_slot != slot:
                raise RuntimeError("another progressive layer is still active")
            state.active = True
            self._active_slot = slot
            return ProgressiveBoundLayer(self, layer, slot)

    def prefetch(self, layer: int) -> bool:
        layer = int(layer)
        with self._cv:
            if layer not in self.layers or layer in self._layer_to_slot:
                self.prefetch_rejected += 1
                return False
            if self._queue:
                self.prefetch_rejected += 1
                return False
            slot = self._choose_slot_locked()
            if slot is None:
                self.prefetch_rejected += 1
                return False
            self._assign_queued_locked(layer, slot)
            self.prefetch_issued += 1
            return True

    def tensor_view(self, bound: ProgressiveBoundLayer, tensor_name: str) -> memoryview:
        if not isinstance(bound, ProgressiveBoundLayer) or bound.reader is not self or bound.released:
            raise ValueError("invalid or released progressive bound layer")
        try:
            meta = self.tensor_index[tensor_name]
        except KeyError as exc:
            raise KeyError(tensor_name) from exc
        if int(meta["layer"]) != bound.layer:
            raise ValueError(f"tensor {tensor_name} is not in bound layer {bound.layer}")
        start = int(meta["offset"])
        end = start + int(meta["nbytes"])
        required = min(int(self.layers[bound.layer]["read_bytes"]), align_up(end, ALIGN))
        waited = False
        t0 = 0.0
        with self._cv:
            state = self._states[bound.slot]
            while state.layer == bound.layer and state.ready_bytes < required and state.status != "error":
                if not waited:
                    waited = True
                    t0 = time.monotonic()
                self._cv.wait()
                state = self._states[bound.slot]
            if waited:
                self.tensor_wait_calls += 1
                self.tensor_wait_seconds += time.monotonic() - t0
            if state.status == "error":
                assert state.error is not None
                raise RuntimeError(f"tensor {tensor_name}: progressive I/O failed") from state.error
            if state.layer != bound.layer or state.ready_bytes < required:
                raise RuntimeError(f"tensor {tensor_name}: readiness state changed unexpectedly")
        return memoryview(self._ring[bound.slot])[start:end]

    def _release_bound(self, bound: ProgressiveBoundLayer) -> None:
        with self._cv:
            state = self._states[bound.slot]
            if state.layer != bound.layer or not state.active:
                raise RuntimeError("progressive bound release state mismatch")
            if state.status != "complete" or state.ready_bytes != state.read_bytes:
                raise RuntimeError("cannot release layer before its full K3 payload is stable")
            state.active = False
            if self._active_slot == bound.slot:
                self._active_slot = None
            self._cv.notify_all()

    def report(self) -> dict[str, Any]:
        with self._cv:
            queued = len(self._queue)
            current = self._current_io_slot
        return {
            "direct_io": self.direct_io,
            "pinned_layers": [],
            "ring_slots": self.plan.ring_slots,
            "slot_bytes": self.plan.slot_bytes,
            "planned_bytes": self.plan.planned_bytes,
            "budget_bytes": self.plan.budget_bytes,
            "bytes_read": self.bytes_read,
            "hits": self.hits,
            "misses": self.misses,
            "async_prefetch_enabled": True,
            "progressive_readiness": True,
            "layout_policy": LAYOUT_POLICY,
            "storage_io_concurrency": 1,
            "queued_requests": queued,
            "io_active": current is not None,
            "max_deferred_layer_requests": 1,
            "max_queued_requests_observed": self.max_queued_requests_observed,
            "pread_calls": self.read_calls,
            "ready_events": self.ready_events,
            "prefetch_issued": self.prefetch_issued,
            "prefetch_rejected": self.prefetch_rejected,
            "tensor_wait_calls": self.tensor_wait_calls,
            "tensor_wait_seconds": self.tensor_wait_seconds,
        }

    def close(self) -> None:
        with self._cv:
            while self._current_io_slot is not None or self._queue:
                self._cv.wait()
            self._stop = True
            self._cv.notify_all()
        self._worker.join()
        try:
            os.close(self.fd)
        finally:
            self.fd = -1
        for buf in self._ring:
            try:
                buf.close()
            except BufferError:
                pass

    def __enter__(self) -> "ProgressiveK3Trunk":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def synthetic_sanity() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        trunk = root / "synthetic.bin"
        manifest_path = root / "synthetic.json"
        blob = bytearray(16 * 1024)
        blob[0:100] = b"A" * 100
        blob[4096:4196] = b"B" * 100
        blob[8192:8292] = b"C" * 100
        blob[12288:12388] = b"D" * 100
        trunk.write_bytes(blob)
        layers = []
        index = {}
        for layer, base, names in ((0, 0, ("a", "b")), (1, 8192, ("a", "b"))):
            tensors = []
            for j, suffix in enumerate(names):
                name = f"blk.{layer}.{suffix}"
                offset = j * 4096
                tensors.append({"name": name, "offset": offset, "nbytes": 100})
                index[name] = {"layer": layer, "offset": offset, "nbytes": 100}
            layers.append({
                "layer": layer,
                "file_offset": base,
                "data_bytes": 4196,
                "read_bytes": 8192,
                "tensor_count": 2,
                "tensors": tensors,
                "readiness_frontiers": [
                    {"stage": "first", "ready_bytes": 4096},
                    {"stage": "last", "ready_bytes": 8192},
                ],
            })
        manifest = {
            "schema": K3_SCHEMA,
            "alignment": ALIGN,
            "tensor_alignment": TENSOR_ALIGN,
            "layout_policy": LAYOUT_POLICY,
            "progressive_readiness": True,
            "layers": layers,
            "tensor_index": index,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with ProgressiveK3Trunk(
            trunk, manifest_path, budget_bytes=16384,
            want_ring=2, max_pinned=0, prefer_direct_io=False) as reader:
            b0 = reader.bind(0)
            if not reader.prefetch(1):
                raise AssertionError("synthetic prefetch was rejected")
            v = reader.tensor_view(b0, "blk.0.a")
            if bytes(v) != b"A" * 100:
                raise AssertionError("synthetic layer0/a mismatch")
            v.release()
            v = reader.tensor_view(b0, "blk.0.b")
            if bytes(v) != b"B" * 100:
                raise AssertionError("synthetic layer0/b mismatch")
            v.release()
            b0.release()
            b1 = reader.bind(1)
            v = reader.tensor_view(b1, "blk.1.a")
            if bytes(v) != b"C" * 100:
                raise AssertionError("synthetic layer1/a mismatch")
            v.release()
            v = reader.tensor_view(b1, "blk.1.b")
            if bytes(v) != b"D" * 100:
                raise AssertionError("synthetic layer1/b mismatch")
            v.release()
            b1.release()
            rep = reader.report()
            if int(rep["bytes_read"]) != 16384 or int(rep["storage_io_concurrency"]) != 1:
                raise AssertionError(f"synthetic report mismatch: {rep}")
            if int(rep["ring_slots"]) != 2 or int(rep["max_queued_requests_observed"]) > 1:
                raise AssertionError(f"synthetic residency/queue mismatch: {rep}")
    print("QWEN38_K3_PROGRESSIVE_SYNTHETIC_PASS")


def real_storage_gate(model: Path, work_dir: Path, output: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    directory = parse_gguf(model)
    trunk = work_dir / "decoder64.progressive.k3.bin"
    manifest_path = work_dir / "decoder64.progressive.k3.json"
    # Use the same pinned metadata consumed by the established Qwen runtime.
    import qwen35_gdn_quant_layer_gate as gdn
    manifest = pack_gguf_layers_progressive(
        directory, trunk, manifest_path, layers=range(N_LAYER),
        model_id=gdn.MODEL_ID, revision=gdn.REVISION,
        source_sha256=gdn.SHA256, expected_layers=N_LAYER)
    if int(manifest["total_read_bytes"]) != K3_STREAM_BYTES:
        raise RuntimeError("progressive packed stream byte total changed")
    max_layer = max(int(x["read_bytes"]) for x in manifest["layers"])
    checked_tensors = 0
    checked_bytes = 0
    with ProgressiveK3Trunk(
        trunk, manifest_path, budget_bytes=2 * max_layer,
        want_ring=2, max_pinned=0, prefer_direct_io=True) as reader:
        for il in range(N_LAYER):
            bound = reader.bind(il)
            if il + 1 < N_LAYER:
                reader.prefetch(il + 1)
            try:
                layer_meta = manifest["layers"][il]
                for tensor in layer_meta["tensors"]:
                    view = reader.tensor_view(bound, str(tensor["name"]))
                    digest = hashlib.sha256(view).hexdigest()
                    nbytes = len(view)
                    view.release()
                    if digest != str(tensor["sha256"]):
                        raise RuntimeError(f"progressive ring tensor hash mismatch: {tensor['name']}")
                    checked_tensors += 1
                    checked_bytes += nbytes
            finally:
                bound.release()
        report = reader.report()
    if not bool(report["direct_io"]):
        raise RuntimeError("real progressive storage gate requires direct I/O")
    if int(report["ring_slots"]) != 2 or int(report["planned_bytes"]) != 672_899_072:
        raise RuntimeError(f"progressive ring budget changed: {report}")
    if int(report["bytes_read"]) != K3_STREAM_BYTES:
        raise RuntimeError(f"progressive reader bytes changed: {report['bytes_read']}")
    if int(report["storage_io_concurrency"]) != 1:
        raise RuntimeError("progressive storage I/O concurrency changed")
    if int(report["max_queued_requests_observed"]) > 1:
        raise RuntimeError("progressive deferred request bound exceeded")
    if checked_tensors != 848:
        raise RuntimeError(f"unexpected decoder tensor count {checked_tensors}")
    result = {
        "schema": "qwen38-k3-progressive-storage-gate-v1",
        "status": "PASS",
        "model_sha256": gdn.SHA256,
        "layout_policy": LAYOUT_POLICY,
        "decoder_tensors_checked": checked_tensors,
        "decoder_tensor_bytes_checked": checked_bytes,
        "k3_read_bytes": K3_STREAM_BYTES,
        "manifest_total_read_bytes": manifest["total_read_bytes"],
        "reader": report,
        "optimization": {
            "tensor_byte_change": False,
            "tensor_order_change_only": True,
            "ring_slots": 2,
            "storage_io_concurrency": 1,
            "max_deferred_layer_requests": 1,
            "arithmetic_change": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("QWEN38_K3_PROGRESSIVE_STORAGE_REAL_PASS")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    rp = sub.add_parser("real")
    rp.add_argument("--model", type=Path, required=True)
    rp.add_argument("--work-dir", type=Path, required=True)
    rp.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "sanity":
        synthetic_sanity()
    else:
        real_storage_gate(args.model, args.work_dir, args.output)


if __name__ == "__main__":
    main()
