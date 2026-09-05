#!/usr/bin/env python3
"""GGUF -> K3 raw quantized layer packer for the Qwen3.8 side lab."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping

from gguf_stream import GGUFDirectory, TensorSpan
from k3_stream import ALIGN, TENSOR_ALIGN, K3Trunk, align_up

LAYER_RE = re.compile(r"^blk\.(\d+)\.")
SOURCE_FORMAT = "gguf-v3"
MANIFEST_VERSION = 2
K3_SCHEMA = "qwen38-k3-trunk-v1"  # reader-compatible; version/source fields disambiguate layout.


def partition_tensors(
    directory: GGUFDirectory, *, expected_layers: int = 64
) -> tuple[dict[int, list[TensorSpan]], dict[int, list[TensorSpan]], list[TensorSpan]]:
    """Partition decoder, auxiliary blk.N (for example MTP), and true globals.

    Qwen3.8 has 64 decoder layers but its GGUF may also encode the one-layer MTP
    head as blk.64.*.  Auxiliary blocks must not be silently treated as decoder
    layers or as global tensors.
    """
    layers = {i: [] for i in range(expected_layers)}
    auxiliary: dict[int, list[TensorSpan]] = {}
    globals_: list[TensorSpan] = []
    for tensor in directory.tensors:
        match = LAYER_RE.match(tensor.name)
        if not match:
            globals_.append(tensor)
            continue
        layer = int(match.group(1))
        if layer in layers:
            layers[layer].append(tensor)
        else:
            auxiliary.setdefault(layer, []).append(tensor)
    missing = [layer for layer, tensors in layers.items() if not tensors]
    if missing:
        raise ValueError(f"GGUF decoder layers missing tensors: {missing}")
    for tensors in layers.values():
        tensors.sort(key=lambda t: t.name)
    for tensors in auxiliary.values():
        tensors.sort(key=lambda t: t.name)
    globals_.sort(key=lambda t: t.name)
    return layers, auxiliary, globals_


def split_tensors(directory: GGUFDirectory, *, expected_layers: int = 64) -> tuple[dict[int, list[TensorSpan]], list[TensorSpan]]:
    """Legacy strict decoder/global split used by existing synthetic gates."""
    layers, auxiliary, globals_ = partition_tensors(directory, expected_layers=expected_layers)
    if auxiliary:
        first = min(auxiliary)
        tensor = auxiliary[first][0]
        raise ValueError(f"GGUF tensor {tensor.name} has out-of-range layer {first}")
    return layers, globals_


def _write_zeros(dst, count: int) -> None:
    zero = b"\0" * min(1024 * 1024, max(1, count))
    while count:
        n = min(count, len(zero))
        dst.write(zero[:n])
        count -= n


def _copy_hash(src_fd: int, source_offset: int, nbytes: int, dst, chunk_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    done = 0
    max_chunk = 0
    while done < nbytes:
        want = min(chunk_bytes, nbytes - done)
        chunk = os.pread(src_fd, want, source_offset + done)
        if len(chunk) != want:
            raise IOError(f"short GGUF read at {source_offset + done}: wanted {want}, got {len(chunk)}")
        dst.write(chunk)
        digest.update(chunk)
        done += want
        max_chunk = max(max_chunk, len(chunk))
    return digest.hexdigest(), max_chunk


def pack_gguf_layers(
    directory: GGUFDirectory,
    out_bin: Path,
    out_index: Path,
    *,
    layers: Iterable[int] | None = None,
    model_id: str,
    revision: str,
    source_sha256: str,
    expected_layers: int = 64,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict:
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
    manifest_layers: list[dict] = []
    total_tensor_bytes = 0
    max_copy_chunk = 0
    src_fd = os.open(directory.path, os.O_RDONLY)
    try:
        with out_bin.open("wb") as dst:
            for layer in selected:
                start = align_up(dst.tell(), ALIGN)
                _write_zeros(dst, start - dst.tell())
                tensors_meta: list[dict] = []
                for tensor in grouped[layer]:
                    rel = align_up(dst.tell() - start, TENSOR_ALIGN)
                    absolute = start + rel
                    _write_zeros(dst, absolute - dst.tell())
                    digest, observed = _copy_hash(src_fd, tensor.data_offset, tensor.nbytes, dst, chunk_bytes)
                    max_copy_chunk = max(max_copy_chunk, observed)
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
                    total_tensor_bytes += tensor.nbytes
                data_bytes = dst.tell() - start
                end = align_up(dst.tell(), ALIGN)
                _write_zeros(dst, end - dst.tell())
                manifest_layers.append({
                    "layer": layer,
                    "file_offset": start,
                    "data_bytes": data_bytes,
                    "read_bytes": end - start,
                    "tensor_count": len(tensors_meta),
                    "tensors": tensors_meta,
                })
    finally:
        os.close(src_fd)

    tensor_index = {
        tensor["name"]: {"layer": layer["layer"], "offset": tensor["offset"], "nbytes": tensor["nbytes"]}
        for layer in manifest_layers for tensor in layer["tensors"]
    }
    manifest = {
        "schema": K3_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "source_format": SOURCE_FORMAT,
        "model_id": model_id,
        "revision": revision,
        "source": {"path": directory.path.name, "sha256": source_sha256, "file_bytes": directory.file_bytes},
        "alignment": ALIGN,
        "tensor_alignment": TENSOR_ALIGN,
        "layers": manifest_layers,
        "auxiliary_blocks": [{
            "block": block,
            "tensor_count": len(tensors),
            "tensors": [{
                "name": t.name, "ggml_type": t.ggml_type, "type_name": t.type_name,
                "shape": list(t.shape), "source_offset": t.data_offset, "nbytes": t.nbytes,
            } for t in tensors],
        } for block, tensors in sorted(auxiliary.items())],
        "globals": [{
            "name": t.name, "ggml_type": t.ggml_type, "type_name": t.type_name,
            "shape": list(t.shape), "source_offset": t.data_offset, "nbytes": t.nbytes,
        } for t in globals_],
        "tensor_index": tensor_index,
        "total_tensor_bytes": total_tensor_bytes,
        "packed_file_bytes": out_bin.stat().st_size,
        "copy_chunk_bytes": chunk_bytes,
        "max_copy_chunk_observed": max_copy_chunk,
    }
    out_index.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class ManifestK3Trunk(K3Trunk):
    """Compatibility alias; base K3Trunk is now manifest-index aware."""
    pass
