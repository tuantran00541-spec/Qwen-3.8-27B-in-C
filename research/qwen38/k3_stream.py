#!/usr/bin/env python3
"""K3-style packed layer streaming primitives for the Qwen3.8 research lane.

This module intentionally stops below semantic inference. It converts local
SafeTensors shards into a layer-contiguous, page-aligned trunk and provides a
small-memory pinned-prefix + ring-slot reader. The design mirrors the useful
storage properties of kimi-k3-in-c while staying model-agnostic enough to gate
Qwen's byte layout before implementing Qwen compute kernels.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ALIGN = 4096
TENSOR_ALIGN = 64
SCHEMA = "qwen38-k3-trunk-v1"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def _product(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        if int(value) < 0:
            raise ValueError("negative tensor dimension")
        out *= int(value)
    return out


@dataclass(frozen=True)
class TensorSpan:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    source_offset: int
    nbytes: int
    layer: int


@dataclass(frozen=True)
class MemoryPlan:
    pinned_layers: tuple[int, ...]
    ring_slots: int
    slot_bytes: int
    planned_bytes: int
    budget_bytes: int


def read_safetensors_header(path: Path) -> dict[str, TensorSpan]:
    """Read only a SafeTensors header and return exact absolute byte spans."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > size - 8:
            raise ValueError(f"{path}: invalid safetensors header length {header_len}")
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(header_raw.decode("utf-8"))
    data_base = 8 + header_len
    spans: dict[str, TensorSpan] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = str(meta["dtype"])
        shape = tuple(int(v) for v in meta["shape"])
        start, end = (int(v) for v in meta["data_offsets"])
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"{path}: unsupported dtype {dtype} for {name}")
        if start < 0 or end < start or data_base + end > size:
            raise ValueError(f"{path}: invalid byte range for {name}")
        expected = _product(shape) * DTYPE_BYTES[dtype]
        if end - start != expected:
            raise ValueError(f"{path}: {name} says {end-start} bytes, shape/dtype imply {expected}")
        match = LAYER_RE.match(name)
        layer = int(match.group(1)) if match else -1
        spans[name] = TensorSpan(
            name=name, shard=path.name, dtype=dtype, shape=shape,
            source_offset=data_base + start, nbytes=end - start, layer=layer,
        )
    return spans


def requested_layer_names(weight_map: Mapping[str, str], layers: Iterable[int]) -> dict[int, list[str]]:
    wanted = {int(layer): [] for layer in layers}
    for name in weight_map:
        match = LAYER_RE.match(name)
        if match:
            layer = int(match.group(1))
            if layer in wanted:
                wanted[layer].append(name)
    missing = [layer for layer, names in wanted.items() if not names]
    if missing:
        raise KeyError(f"no language-layer tensors found for layers {missing}")
    for names in wanted.values():
        names.sort()
    return wanted


def needed_shards(weight_map: Mapping[str, str], layers: Iterable[int]) -> list[str]:
    names = requested_layer_names(weight_map, layers)
    return sorted({str(weight_map[name]) for layer_names in names.values() for name in layer_names})


def collect_local_spans(model_dir: Path, weight_map: Mapping[str, str], layers: Iterable[int]) -> dict[int, list[TensorSpan]]:
    requested = requested_layer_names(weight_map, layers)
    shard_names = sorted({weight_map[name] for names in requested.values() for name in names})
    by_name: dict[str, TensorSpan] = {}
    for shard in shard_names:
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        by_name.update(read_safetensors_header(path))
    out: dict[int, list[TensorSpan]] = {}
    for layer, names in requested.items():
        spans: list[TensorSpan] = []
        for name in names:
            span = by_name.get(name)
            if span is None:
                raise KeyError(f"{name} missing from downloaded safetensors headers")
            expected_shard = str(weight_map[name])
            if span.shard != expected_shard:
                raise ValueError(f"{name}: index={expected_shard}, header={span.shard}")
            spans.append(span)
        out[layer] = spans
    return out


def _write_zeros(handle, count: int) -> None:
    zero = b"\0" * min(1024 * 1024, max(1, count))
    left = count
    while left:
        chunk = min(left, len(zero))
        handle.write(zero[:chunk])
        left -= chunk


def _copy_range_with_hash(src: Path, source_offset: int, nbytes: int, dst, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    fd = os.open(src, os.O_RDONLY)
    try:
        done = 0
        while done < nbytes:
            want = min(chunk_bytes, nbytes - done)
            chunk = os.pread(fd, want, source_offset + done)
            if len(chunk) != want:
                raise IOError(f"short read from {src}: wanted {want}, got {len(chunk)}")
            dst.write(chunk)
            digest.update(chunk)
            done += want
    finally:
        os.close(fd)
    return digest.hexdigest()


def pack_layers(
    model_dir: Path,
    weight_map: Mapping[str, str],
    layers: Iterable[int],
    out_bin: Path,
    out_index: Path,
    *,
    model_id: str,
    revision: str,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict:
    """Pack requested language layers into one contiguous aligned trunk.

    Source shards stay untouched. Tensor data is copied in bounded chunks; no
    whole weight tensor is materialized in Python memory.
    """
    layers = sorted({int(layer) for layer in layers})
    if not layers:
        raise ValueError("at least one layer is required")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    spans = collect_local_spans(model_dir, weight_map, layers)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    manifest_layers: list[dict] = []
    total_data = 0
    with out_bin.open("wb") as dst:
        for layer in layers:
            start = align_up(dst.tell(), ALIGN)
            if start > dst.tell():
                _write_zeros(dst, start - dst.tell())
            tensors: list[dict] = []
            for span in spans[layer]:
                tensor_at = align_up(dst.tell() - start, TENSOR_ALIGN)
                absolute = start + tensor_at
                if absolute > dst.tell():
                    _write_zeros(dst, absolute - dst.tell())
                sha256 = _copy_range_with_hash(
                    model_dir / span.shard, span.source_offset, span.nbytes, dst, chunk_bytes
                )
                tensors.append({
                    "name": span.name,
                    "dtype": span.dtype,
                    "shape": list(span.shape),
                    "offset": tensor_at,
                    "nbytes": span.nbytes,
                    "source_shard": span.shard,
                    "source_offset": span.source_offset,
                    "sha256": sha256,
                })
                total_data += span.nbytes
            data_bytes = dst.tell() - start
            end = align_up(dst.tell(), ALIGN)
            if end > dst.tell():
                _write_zeros(dst, end - dst.tell())
            manifest_layers.append({
                "layer": layer,
                "file_offset": start,
                "data_bytes": data_bytes,
                "read_bytes": end - start,
                "tensor_count": len(tensors),
                "tensors": tensors,
            })

    manifest = {
        "schema": SCHEMA,
        "model_id": model_id,
        "revision": revision,
        "alignment": ALIGN,
        "tensor_alignment": TENSOR_ALIGN,
        "layers": manifest_layers,
        "total_tensor_bytes": total_data,
        "packed_file_bytes": out_bin.stat().st_size,
    }
    out_index.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _layer_map(manifest: Mapping) -> dict[int, dict]:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported trunk schema: {manifest.get('schema')!r}")
    if int(manifest.get("alignment", 0)) != ALIGN:
        raise ValueError("trunk alignment mismatch")
    out: dict[int, dict] = {}
    for raw in manifest.get("layers", []):
        layer = int(raw["layer"])
        if layer in out:
            raise ValueError(f"duplicate layer {layer}")
        offset = int(raw["file_offset"])
        read_bytes = int(raw["read_bytes"])
        data_bytes = int(raw["data_bytes"])
        if offset % ALIGN or read_bytes % ALIGN or not (0 <= data_bytes <= read_bytes):
            raise ValueError(f"layer {layer}: invalid aligned range")
        out[layer] = dict(raw)
    if not out:
        raise ValueError("trunk contains no layers")
    return out


def _build_tensor_index(manifest: Mapping, layers: Mapping[int, Mapping]) -> dict[str, dict]:
    """Return a validated name -> layer-relative byte span index.

    GGUF manifests provide a top-level ``tensor_index`` because tensor names do
    not follow the legacy Hugging Face regex.  Older SafeTensors manifests do
    not, so derive the same index from ``layers[].tensors``.  This keeps tensor
    lookup manifest-driven without breaking the already-proven SafeTensors K3
    path.
    """
    raw = manifest.get("tensor_index")
    if raw is None:
        raw = {
            str(tensor["name"]): {
                "layer": int(layer),
                "offset": int(tensor["offset"]),
                "nbytes": int(tensor["nbytes"]),
            }
            for layer, layer_meta in layers.items()
            for tensor in layer_meta.get("tensors", [])
        }
    if not isinstance(raw, Mapping):
        raise ValueError("tensor_index must be a mapping")

    out: dict[str, dict] = {}
    for raw_name, raw_meta in raw.items():
        name = str(raw_name)
        if not name or name in out:
            raise ValueError(f"invalid or duplicate tensor index name: {name!r}")
        if not isinstance(raw_meta, Mapping):
            raise ValueError(f"tensor {name}: invalid tensor index entry")
        layer = int(raw_meta["layer"])
        offset = int(raw_meta["offset"])
        nbytes = int(raw_meta["nbytes"])
        layer_meta = layers.get(layer)
        if layer_meta is None:
            raise ValueError(f"tensor {name}: unknown layer {layer}")
        read_bytes = int(layer_meta["read_bytes"])
        if offset < 0 or nbytes < 0 or offset + nbytes > read_bytes:
            raise ValueError(f"tensor {name}: byte span exceeds layer {layer}")
        out[name] = {"layer": layer, "offset": offset, "nbytes": nbytes}
    return out


def plan_memory(manifest: Mapping, budget_bytes: int, *, want_ring: int = 2, max_pinned: int | None = None) -> MemoryPlan:
    """Choose the largest pinned prefix while preferring two safe ring slots."""
    layers = _layer_map(manifest)
    order = sorted(layers)
    budget_bytes = int(budget_bytes)
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if want_ring not in (1, 2):
        raise ValueError("want_ring must be 1 or 2")
    pin_cap = len(order) if max_pinned is None else max(0, min(len(order), int(max_pinned)))

    def best_for_ring(ring_slots: int) -> MemoryPlan | None:
        best: MemoryPlan | None = None
        for npin in range(pin_cap + 1):
            pinned = order[:npin]
            streamed = order[npin:]
            pin_bytes = sum(int(layers[layer]["read_bytes"]) for layer in pinned)
            if streamed:
                slot_bytes = max(int(layers[layer]["read_bytes"]) for layer in streamed)
                planned = pin_bytes + ring_slots * slot_bytes
                actual_ring = ring_slots
            else:
                slot_bytes = 0
                planned = pin_bytes
                actual_ring = 0
            if planned <= budget_bytes:
                best = MemoryPlan(tuple(pinned), actual_ring, slot_bytes, planned, budget_bytes)
        return best

    for ring in range(want_ring, 0, -1):
        plan = best_for_ring(ring)
        if plan is not None:
            return plan
    raise MemoryError("budget cannot hold even one streaming layer slot")


class K3Trunk:
    """Pinned-prefix + ring-slot reader for a qwen38-k3-trunk-v1 file.

    Tensor lookup is manifest-driven.  GGUF manifests can provide a top-level
    ``tensor_index``; legacy SafeTensors manifests are indexed from their layer
    tensor records.  Async prefetch is correctness-gated: it is disabled when
    fewer than two ring slots exist, so a worker can never overwrite the layer
    currently bound for compute.
    """

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
        self.layers = _layer_map(self.manifest)
        self.tensor_index = _build_tensor_index(self.manifest, self.layers)
        self.order = sorted(self.layers)
        self.plan = plan_memory(self.manifest, budget_bytes, want_ring=want_ring, max_pinned=max_pinned)
        self._io_lock = threading.Lock()
        self.direct_io = False
        self.fd = self._open_fd(prefer_direct_io)
        size = self.bin_path.stat().st_size
        for layer, meta in self.layers.items():
            if int(meta["file_offset"]) + int(meta["read_bytes"]) > size:
                raise ValueError(f"layer {layer}: trunk range exceeds file size")

        self._pinned: dict[int, mmap.mmap] = {}
        self._pinned_loaded: set[int] = set()
        self._ring: list[mmap.mmap] = [mmap.mmap(-1, self.plan.slot_bytes) for _ in range(self.plan.ring_slots)]
        self._layer_of = [-1] * self.plan.ring_slots
        self._slot_of: dict[int, int] = {}
        self._ring_cursor = 0
        self._active_slot: int | None = None
        self._executor = ThreadPoolExecutor(max_workers=1) if self.plan.ring_slots >= 2 else None
        self._pending: tuple[int, int, Future[int]] | None = None
        self.bytes_read = 0
        self.hits = 0
        self.misses = 0

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
        with self._io_lock:
            if not self.direct_io:
                return
            os.close(self.fd)
            self.fd = os.open(self.bin_path, os.O_RDONLY)
            self.direct_io = False

    def _pread_into(self, target: mmap.mmap, nbytes: int, offset: int) -> int:
        if nbytes % ALIGN or offset % ALIGN:
            raise ValueError("direct-read ranges must be page aligned")
        view = memoryview(target)[:nbytes]
        done = 0
        while done < nbytes:
            try:
                if hasattr(os, "preadv"):
                    got = os.preadv(self.fd, [view[done:]], offset + done)
                else:
                    chunk = os.pread(self.fd, nbytes - done, offset + done)
                    got = len(chunk)
                    view[done : done + got] = chunk
            except OSError as exc:
                if self.direct_io and exc.errno in (22, 95):
                    self._fallback_buffered()
                    continue
                raise
            if got <= 0:
                raise IOError(f"short read at {offset + done}")
            done += got
        view.release()
        self.bytes_read += done
        return done

    def _load_layer(self, layer: int, target: mmap.mmap) -> int:
        meta = self.layers[layer]
        return self._pread_into(target, int(meta["read_bytes"]), int(meta["file_offset"]))

    def _finish_pending(self, requested_layer: int | None = None) -> bool:
        pending = self._pending
        if pending is None:
            return False
        layer, slot, future = pending
        if requested_layer is not None and layer != requested_layer:
            return False
        future.result()
        self._layer_of[slot] = layer
        self._slot_of[layer] = slot
        self._pending = None
        return True

    def bind(self, layer: int) -> memoryview:
        layer = int(layer)
        if layer not in self.layers:
            raise KeyError(layer)
        meta = self.layers[layer]
        read_bytes = int(meta["read_bytes"])
        if layer in self.plan.pinned_layers:
            buf = self._pinned.get(layer)
            if buf is None:
                buf = mmap.mmap(-1, read_bytes)
                self._pinned[layer] = buf
            if layer not in self._pinned_loaded:
                self._load_layer(layer, buf)
                self._pinned_loaded.add(layer)
                self.misses += 1
            else:
                self.hits += 1
            self._active_slot = None
            return memoryview(buf)[:read_bytes]

        if self._finish_pending(layer):
            slot = self._slot_of[layer]
            self.hits += 1
            self._active_slot = slot
            return memoryview(self._ring[slot])[:read_bytes]

        slot = self._slot_of.get(layer)
        if slot is not None and self._layer_of[slot] == layer:
            self.hits += 1
            self._active_slot = slot
            return memoryview(self._ring[slot])[:read_bytes]

        self._finish_pending()
        if not self._ring:
            raise RuntimeError("all layers are pinned; streamed bind should be unreachable")
        slot = self._ring_cursor
        self._ring_cursor = (self._ring_cursor + 1) % len(self._ring)
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
        if self._executor is None or self.plan.ring_slots < 2:
            return False
        if layer not in self.layers or layer in self.plan.pinned_layers or layer in self._slot_of:
            return False
        if self._pending is not None:
            return False
        slot = self._ring_cursor
        if self._active_slot is not None and slot == self._active_slot:
            slot = (slot + 1) % len(self._ring)
        if self._active_slot is not None and slot == self._active_slot:
            return False
        self._ring_cursor = (slot + 1) % len(self._ring)
        old = self._layer_of[slot]
        if old >= 0:
            self._slot_of.pop(old, None)
        self._layer_of[slot] = -1
        future = self._executor.submit(self._load_layer, layer, self._ring[slot])
        self._pending = (layer, slot, future)
        return True

    def tensor_view(self, layer_view: memoryview, tensor_name: str) -> memoryview:
        try:
            meta = self.tensor_index[tensor_name]
        except KeyError as exc:
            raise KeyError(tensor_name) from exc
        start = int(meta["offset"])
        end = start + int(meta["nbytes"])
        if start < 0 or end > len(layer_view):
            raise ValueError(f"tensor {tensor_name} exceeds bound layer view")
        return layer_view[start:end]

    def report(self) -> dict:
        return {
            "direct_io": self.direct_io,
            "pinned_layers": list(self.plan.pinned_layers),
            "ring_slots": self.plan.ring_slots,
            "slot_bytes": self.plan.slot_bytes,
            "planned_bytes": self.plan.planned_bytes,
            "budget_bytes": self.plan.budget_bytes,
            "bytes_read": self.bytes_read,
            "hits": self.hits,
            "misses": self.misses,
            "async_prefetch_enabled": self._executor is not None,
        }

    def close(self) -> None:
        if self._pending is not None:
            self._finish_pending()
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

    def __enter__(self) -> "K3Trunk":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def verify_layer(manifest: Mapping, trunk_path: Path, layer: int) -> dict:
    layers = _layer_map(manifest)
    meta = layers[int(layer)]
    fd = os.open(trunk_path, os.O_RDONLY)
    checked = 0
    checked_bytes = 0
    try:
        for tensor in meta["tensors"]:
            left = int(tensor["nbytes"])
            offset = int(meta["file_offset"]) + int(tensor["offset"])
            digest = hashlib.sha256()
            done = 0
            while done < left:
                want = min(8 * 1024 * 1024, left - done)
                chunk = os.pread(fd, want, offset + done)
                if len(chunk) != want:
                    raise IOError("short packed verification read")
                digest.update(chunk)
                done += want
            if digest.hexdigest() != tensor["sha256"]:
                raise ValueError(f"packed checksum mismatch: {tensor['name']}")
            checked += 1
            checked_bytes += left
    finally:
        os.close(fd)
    return {"layer": int(layer), "tensors_checked": checked, "tensor_bytes_checked": checked_bytes}
