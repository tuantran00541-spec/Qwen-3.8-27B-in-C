#!/usr/bin/env python3
"""Native-Windows full-real exact current-best Qwen3.8 profile adapter.

The proven decoder/profile is reused rather than forked.  This adapter changes
only platform plumbing while keeping the Linux evidence path untouched:

* install the existing Win32 resource/UCRT bootstrap before legacy imports;
* provide a locked binary seek/read ``os.pread`` compatibility for the legacy
  small positional GGUF reads used by output-norm and token embeddings;
* pack the execution-ordered K3 trunk with an explicit-offset seek/read copy
  because Python on Windows has no os.pread;
* bind the proven Win32 NO_BUFFERING|OVERLAPPED progressive two-slot reader;
* bind the portable exact two-worker Q6 current-best stack.

Decoder arithmetic, tensor bytes, execution order, ring residency, storage I/O
concurrency, Q6 row partitioning, and all hidden/state anchors remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any

from qwen38_win32_bootstrap import import_generator, install_resource_compat

if sys.platform == "win32":
    install_resource_compat()
    gen = import_generator()
else:  # Keep import diagnostics readable; real execution is Windows-only.
    gen = None

import qwen38_k3_progressive as progressive
from qwen38_k3_progressive_win32 import Win32ProgressiveK3Trunk
from qwen38_current_best_runtime_win32 import (
    Qwen38CurrentBestWin32QuantStack,
    sanity as win32_quant_sanity,
)

# Import after resource compatibility is installed.  This module and its
# transitive evidence modules intentionally retain their proven `import resource`.
import qwen38_current_best_q6_profile as base

K3_STREAM_BYTES = 21_127_430_144
K3_RING_SLOTS = 2
K3_PLANNED_BYTES = 672_899_072
K3_STORAGE_IO_CONCURRENCY = 1
K3_MAX_DEFERRED_LAYER_REQUESTS = 1
_PACK_LOCK = threading.Lock()
_PREAD_LOCK = threading.Lock()


def _pread_seek_read(fd: int, nbytes: int, offset: int) -> bytes:
    """Win32 ``os.pread`` compatibility for independently-opened GGUF fds."""
    if sys.platform != "win32":
        raise RuntimeError("Win32 pread compatibility requires native Windows")
    if int(nbytes) < 0 or int(offset) < 0:
        raise ValueError("pread nbytes/offset must be non-negative")
    import msvcrt

    with _PREAD_LOCK:
        msvcrt.setmode(fd, os.O_BINARY)
        restore = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, int(offset), os.SEEK_SET)
            chunks: list[bytes] = []
            done = 0
            while done < int(nbytes):
                chunk = os.read(fd, int(nbytes) - done)
                if not chunk:
                    break
                chunks.append(chunk)
                done += len(chunk)
            return b"".join(chunks)
        finally:
            os.lseek(fd, restore, os.SEEK_SET)


def _install_pread_compat() -> None:
    """Install the local Win32 positional-read shim only when Python lacks it."""
    if hasattr(os, "pread"):
        return
    if sys.platform != "win32":
        raise RuntimeError("os.pread unexpectedly missing on non-Windows host")
    setattr(os, "pread", _pread_seek_read)


def _copy_hash_seek_read(
    src_fd: int,
    source_offset: int,
    nbytes: int,
    dst,
    chunk_bytes: int,
) -> tuple[str, int]:
    """Windows-compatible exact-offset GGUF copy used only while packing K3."""
    if sys.platform != "win32":
        raise RuntimeError("Win32 GGUF pack copy requires native Windows")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    import msvcrt
    msvcrt.setmode(src_fd, os.O_BINARY)

    digest = hashlib.sha256()
    done = 0
    max_chunk = 0
    while done < int(nbytes):
        want = min(int(chunk_bytes), int(nbytes) - done)
        os.lseek(src_fd, int(source_offset) + done, os.SEEK_SET)
        chunk = os.read(src_fd, want)
        if len(chunk) != want:
            raise IOError(
                f"short Win32 GGUF read at {int(source_offset) + done}: "
                f"wanted {want}, got {len(chunk)}")
        dst.write(chunk)
        digest.update(chunk)
        done += want
        max_chunk = max(max_chunk, len(chunk))
    return digest.hexdigest(), max_chunk


def pack_gguf_layers_progressive_win32(*args, **kwargs):
    """Call the proven progressive packer with only its source-read primitive replaced."""
    if sys.platform != "win32":
        raise RuntimeError("Win32 progressive packer requires native Windows")
    with _PACK_LOCK:
        original = progressive._copy_hash
        progressive._copy_hash = _copy_hash_seek_read
        try:
            manifest = progressive.pack_gguf_layers_progressive(*args, **kwargs)
        finally:
            progressive._copy_hash = original
    layers = manifest.get("layers", [])
    if len(layers) == 64 and int(manifest.get("total_read_bytes", -1)) != K3_STREAM_BYTES:
        raise RuntimeError(
            f"Win32 progressive pack changed K3 bytes: "
            f"{manifest.get('total_read_bytes')} != {K3_STREAM_BYTES}")
    return manifest


def _reader_factory(win32_direct_lib: Path):
    def build(
        bin_path: Path,
        index_path: Path,
        *,
        budget_bytes: int,
        want_ring: int = 2,
        max_pinned: int | None = None,
        prefer_direct_io: bool = True,
    ):
        if not prefer_direct_io:
            raise RuntimeError("Win32 current-best requires native direct I/O")
        return Win32ProgressiveK3Trunk(
            bin_path,
            index_path,
            budget_bytes=int(budget_bytes),
            win32_direct_lib=Path(win32_direct_lib),
            want_ring=int(want_ring),
            max_pinned=max_pinned,
        )
    return build


def _copy_sanity() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Win32 copy sanity requires native Windows")
    payload = bytes(((i * 73 + 19) & 0xFF) for i in range(131_317))
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "source.bin"
        dst_path = root / "copy.bin"
        prefix = b"prefix\r\n" * 113
        suffix = b"suffix\n" * 29
        src.write_bytes(prefix + payload + suffix)
        fd = os.open(src, os.O_RDONLY)
        try:
            with dst_path.open("wb") as dst:
                digest, observed = _copy_hash_seek_read(
                    fd, len(prefix), len(payload), dst, 4093)
        finally:
            os.close(fd)
        if dst_path.read_bytes() != payload:
            raise RuntimeError("Win32 seek/read copy changed bytes")
        if digest != hashlib.sha256(payload).hexdigest():
            raise RuntimeError("Win32 seek/read copy SHA mismatch")
        if observed > 4093:
            raise RuntimeError("Win32 seek/read copy exceeded chunk contract")


def _pread_sanity() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Win32 pread sanity requires native Windows")
    _install_pread_compat()
    prefix = b"prefix\x1a\r\n" * 17
    payload = bytes(((i * 29 + 7) & 0xFF) for i in range(8193))
    suffix = b"\ntrailer\x1a" * 11
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pread.bin"
        path.write_bytes(prefix + payload + suffix)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.lseek(fd, 5, os.SEEK_SET)
            before = os.lseek(fd, 0, os.SEEK_CUR)
            raw = os.pread(fd, len(payload), len(prefix))
            after = os.lseek(fd, 0, os.SEEK_CUR)
        finally:
            os.close(fd)
    if raw != payload:
        raise RuntimeError("Win32 pread compatibility changed bytes")
    if before != after:
        raise RuntimeError("Win32 pread compatibility changed file offset")


def sanity() -> None:
    if sys.platform != "win32":
        raise SystemExit("Win32 current-best profile sanity must run on Windows")
    if gen is None or int(gen.N_LAYER) != 64:
        raise SystemExit("Win32 generator bootstrap did not expose 64 layers")
    win32_quant_sanity()
    _copy_sanity()
    _pread_sanity()
    if K3_RING_SLOTS != 2 or K3_PLANNED_BYTES != 672_899_072:
        raise SystemExit("Win32 current-best K3 residency contract changed")
    if K3_STORAGE_IO_CONCURRENCY != 1 or K3_MAX_DEFERRED_LAYER_REQUESTS != 1:
        raise SystemExit("Win32 current-best K3 queue contract changed")
    if K3_STREAM_BYTES != base.K3_STREAM_BYTES:
        raise SystemExit("Win32 current-best K3 stream-byte contract changed")
    print("QWEN38_WIN32_PROGRESSIVE_CURRENT_BEST_SANITY PASS")


def run(args) -> dict[str, Any]:
    if sys.platform != "win32" or gen is None:
        raise RuntimeError("full real Win32 current-best gate requires native Windows")
    _install_pread_compat()

    old_pack = gen.pack_gguf_layers
    old_reader = gen.K3Trunk
    old_stack = base.Qwen38CurrentBestQuantStack
    gen.pack_gguf_layers = pack_gguf_layers_progressive_win32
    gen.K3Trunk = _reader_factory(args.win32_direct_lib)
    base.Qwen38CurrentBestQuantStack = Qwen38CurrentBestWin32QuantStack
    try:
        payload = base.run(args)
    finally:
        base.Qwen38CurrentBestQuantStack = old_stack
        gen.K3Trunk = old_reader
        gen.pack_gguf_layers = old_pack

    reader = payload["reader"]
    if reader.get("platform") != "win32":
        raise RuntimeError(f"full real profile did not use Win32 reader: {reader}")
    if not bool(reader.get("direct_io")):
        raise RuntimeError("full real Win32 profile did not use direct I/O")
    if not bool(reader.get("win32_no_buffering")) or not bool(reader.get("win32_overlapped")):
        raise RuntimeError("full real Win32 profile lost NO_BUFFERING/OVERLAPPED")
    if int(reader.get("ring_slots", -1)) != K3_RING_SLOTS:
        raise RuntimeError("full real Win32 profile changed ring slots")
    if int(reader.get("native_ring_slots", -1)) != K3_RING_SLOTS:
        raise RuntimeError("full real Win32 profile changed native ring slots")
    if int(reader.get("planned_bytes", -1)) != K3_PLANNED_BYTES:
        raise RuntimeError("full real Win32 profile changed planned ring bytes")
    if int(reader.get("bytes_read", -1)) != K3_STREAM_BYTES:
        raise RuntimeError("full real Win32 profile changed K3 stream bytes")
    if int(reader.get("win32_native_read_bytes", -1)) != K3_STREAM_BYTES:
        raise RuntimeError("native Win32 byte count differs from exact K3 stream")
    if int(reader.get("storage_io_concurrency", -1)) != K3_STORAGE_IO_CONCURRENCY:
        raise RuntimeError("full real Win32 profile changed storage I/O concurrency")
    if int(reader.get("max_deferred_layer_requests", -1)) != K3_MAX_DEFERRED_LAYER_REQUESTS:
        raise RuntimeError("full real Win32 profile changed deferred-layer contract")
    if int(reader.get("max_queued_requests_observed", 99)) > K3_MAX_DEFERRED_LAYER_REQUESTS:
        raise RuntimeError("full real Win32 profile exceeded deferred-layer contract")

    quant = payload["quant_stack"]
    if quant.get("platform") != "win32":
        raise RuntimeError(f"full real profile did not use Win32 quant stack: {quant}")
    if int(quant.get("q6_workers", -1)) != 2:
        raise RuntimeError("full real Win32 profile changed Q6 worker count")
    if not bool(quant.get("q8_noalloc")):
        raise RuntimeError("full real Win32 profile lost Q8 noalloc")

    payload["schema"] = "qwen38-win32-progressive-current-best-full-real-v1"
    payload["claim"] = (
        "native Windows hosted full-real exact 11-token decoder-prefill gate using "
        "portable Q6-pool2 plus execution-ordered Win32 NO_BUFFERING/OVERLAPPED progressive K3")
    payload["win32_full_real"] = {
        "model_loaded": True,
        "model_sha256": payload["model_sha256"],
        "k3_pack_source_read": "binary os.lseek+os.read explicit-offset compatibility",
        "runtime_storage": "CreateFileW NO_BUFFERING|OVERLAPPED progressive two-slot ring",
        "ring_slots": K3_RING_SLOTS,
        "planned_bytes": K3_PLANNED_BYTES,
        "storage_io_concurrency": K3_STORAGE_IO_CONCURRENCY,
        "max_deferred_layer_requests": K3_MAX_DEFERRED_LAYER_REQUESTS,
        "arithmetic_change": False,
        "tensor_byte_change": False,
        "linux_runtime_changed": False,
    }
    payload["optimization"].update({
        "linux_experimental": False,
        "windows_full_real_gate": True,
        "windows_backend_promoted": False,
        "reader_policy_change": True,
        "execution_ordered_k3_layout": True,
        "progressive_tensor_readiness": True,
        "ring_change": False,
        "arithmetic_change": False,
    })
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "prefill_seconds": payload["prefill_seconds"],
        "hidden_sha256": payload["hidden_sha256"],
        "state_sha256": payload["state_sha256"],
        "k3_bytes": payload["k3_bytes"],
        "q6_boundary_seconds": payload["q6_boundary_seconds"],
        "reader_bind_seconds": payload["reader_bind_seconds"],
        "reader": reader,
        "quant_stack": quant,
        "max_rss_gib": payload["max_rss_gib"],
    }, indent=2, ensure_ascii=False))
    print("QWEN38_WIN32_FULL_GGUF_CURRENT_BEST_REAL_BITWISE_PASS")
    return payload


def parser() -> argparse.ArgumentParser:
    ap = base.parser()
    subparsers = next(
        action for action in ap._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    subparsers.choices["run"].add_argument(
        "--win32-direct-lib", type=Path, required=True)
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
