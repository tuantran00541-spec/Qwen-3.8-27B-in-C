#!/usr/bin/env python3
"""Measurement-only K3 intra-layer Direct-I/O queue-depth sweep.

This probe does not change the inference runtime.  It packs the pinned GGUF into
the same 64-layer K3 trunk layout, allocates the same two-slot ring budget, and
compares the current one-shot preadv() layer read against chunked concurrent
preadv() reads that target disjoint ranges of the *same* ring slot.

The experiment varies only storage request granularity/queue depth.  Model
residency, ring slots, packed bytes, tensor layout, and arithmetic are unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import Any

from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import ALIGN, K3Trunk

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GGUF_SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
EXPECTED_K3_BYTES = 21_127_430_144
N_LAYER = 64

# Balanced sample of recurrent and full-attention layers across the decoder.
SAMPLE_LAYERS = (0, 3, 4, 7, 16, 19, 20, 23, 32, 35, 36, 39, 48, 51, 60, 63)
BLOCK_SIZES = (1 * 1024 * 1024, 4 * 1024 * 1024)
QUEUE_DEPTHS = (1, 2, 4, 8)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _sha256_view(buf: mmap.mmap, nbytes: int) -> str:
    view = memoryview(buf)[:nbytes]
    try:
        return hashlib.sha256(view).hexdigest()
    finally:
        view.release()


def _preadv_exact(fd: int, target: mmap.mmap, start: int, nbytes: int, file_offset: int) -> int:
    view = memoryview(target)[start : start + nbytes]
    try:
        done = 0
        while done < nbytes:
            got = os.preadv(fd, [view[done:]], file_offset + done)
            if got <= 0:
                raise IOError(f"short read at {file_offset + done}")
            done += int(got)
        return done
    finally:
        view.release()


def _read_one_shot(fd: int, target: mmap.mmap, nbytes: int, file_offset: int) -> tuple[int, int]:
    got = _preadv_exact(fd, target, 0, nbytes, file_offset)
    return got, 1


def _chunk_ranges(nbytes: int, block_bytes: int) -> list[tuple[int, int]]:
    if nbytes % ALIGN or block_bytes % ALIGN:
        raise ValueError("direct-I/O ranges must remain page aligned")
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    out: list[tuple[int, int]] = []
    start = 0
    while start < nbytes:
        size = min(block_bytes, nbytes - start)
        if size % ALIGN:
            raise ValueError("final direct-I/O chunk is not page aligned")
        out.append((start, size))
        start += size
    return out


def _read_chunked(
    fd: int,
    target: mmap.mmap,
    nbytes: int,
    file_offset: int,
    block_bytes: int,
    queue_depth: int,
    executor: ThreadPoolExecutor | None,
) -> tuple[int, int]:
    chunks = _chunk_ranges(nbytes, block_bytes)
    if queue_depth == 1:
        done = 0
        for start, size in chunks:
            done += _preadv_exact(fd, target, start, size, file_offset + start)
        return done, len(chunks)
    if executor is None:
        raise RuntimeError("queue_depth > 1 requires a persistent executor")
    futures = [
        executor.submit(_preadv_exact, fd, target, start, size, file_offset + start)
        for start, size in chunks
    ]
    done = sum(int(f.result()) for f in futures)
    return done, len(chunks)


def sanity() -> None:
    if not hasattr(os, "preadv"):
        raise SystemExit("queue-depth probe requires os.preadv")
    if any(x % ALIGN for x in BLOCK_SIZES):
        raise SystemExit("block sizes must be K3-page aligned")
    if any(q < 1 for q in QUEUE_DEPTHS):
        raise SystemExit("queue depths must be positive")
    if len(set(SAMPLE_LAYERS)) != len(SAMPLE_LAYERS):
        raise SystemExit("sample layer list contains duplicates")
    if min(SAMPLE_LAYERS) < 0 or max(SAMPLE_LAYERS) >= N_LAYER:
        raise SystemExit("sample layer outside decoder range")
    print("QWEN38_K3_IO_QD_SANITY PASS")


def synthetic() -> None:
    sanity()
    total = 9 * 1024 * 1024
    payload = bytes((i * 37 + 11) & 0xFF for i in range(256)) * (total // 256)
    with tempfile.TemporaryDirectory(prefix="qwen38-k3-qd-") as td:
        path = Path(td) / "synthetic.bin"
        path.write_bytes(payload)
        fd = os.open(path, os.O_RDONLY)
        target = mmap.mmap(-1, total)
        expected = hashlib.sha256(payload).hexdigest()
        try:
            got, _ = _read_one_shot(fd, target, total, 0)
            if got != total or _sha256_view(target, total) != expected:
                raise SystemExit("synthetic one-shot mismatch")
            for block_bytes in BLOCK_SIZES:
                for qd in QUEUE_DEPTHS:
                    executor = ThreadPoolExecutor(max_workers=qd) if qd > 1 else None
                    try:
                        got, _ = _read_chunked(
                            fd, target, total, 0, block_bytes, qd, executor
                        )
                    finally:
                        if executor is not None:
                            executor.shutdown(wait=True)
                    if got != total or _sha256_view(target, total) != expected:
                        raise SystemExit(
                            f"synthetic chunk mismatch block={block_bytes} qd={qd}"
                        )
        finally:
            target.close()
            os.close(fd)
    print("QWEN38_K3_IO_QD_SYNTHETIC_PASS")


def _method_name(block_bytes: int | None, queue_depth: int | None) -> str:
    if block_bytes is None:
        return "one_shot_current"
    return f"chunk_{block_bytes // (1024 * 1024)}m_qd{queue_depth}"


def _run_method(
    *,
    fd: int,
    target: mmap.mmap,
    layer_meta: dict[int, dict[str, Any]],
    layers: tuple[int, ...],
    block_bytes: int | None,
    queue_depth: int | None,
    expected_sha: dict[int, str],
) -> dict[str, Any]:
    name = _method_name(block_bytes, queue_depth)
    executor = (
        ThreadPoolExecutor(max_workers=int(queue_depth))
        if block_bytes is not None and int(queue_depth) > 1
        else None
    )
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    total_bytes = 0
    total_calls = 0
    try:
        for layer in layers:
            meta = layer_meta[layer]
            nbytes = int(meta["read_bytes"])
            file_offset = int(meta["file_offset"])
            t0 = time.monotonic()
            if block_bytes is None:
                got, calls = _read_one_shot(fd, target, nbytes, file_offset)
            else:
                got, calls = _read_chunked(
                    fd,
                    target,
                    nbytes,
                    file_offset,
                    int(block_bytes),
                    int(queue_depth),
                    executor,
                )
            seconds = time.monotonic() - t0
            if got != nbytes:
                raise RuntimeError(f"{name} layer {layer}: read {got} != {nbytes}")
            digest = _sha256_view(target, nbytes)
            old = expected_sha.get(layer)
            if old is None:
                if block_bytes is not None:
                    raise RuntimeError(
                        f"{name} layer {layer}: candidate ran before reference digest existed"
                    )
                expected_sha[layer] = digest
            elif digest != old:
                raise RuntimeError(
                    f"{name} layer {layer}: byte mismatch {digest} != {old}"
                )
            total_seconds += seconds
            total_bytes += got
            total_calls += calls
            rows.append(
                {
                    "layer": layer,
                    "read_bytes": got,
                    "seconds": seconds,
                    "gib_s": (got / (1024.0 ** 3)) / seconds if seconds > 0 else None,
                    "preadv_calls": calls,
                    "sha256": digest,
                }
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return {
        "name": name,
        "block_bytes": block_bytes,
        "queue_depth": queue_depth,
        "layers": rows,
        "read_bytes": total_bytes,
        "seconds": total_seconds,
        "preadv_calls": total_calls,
        "gib_s": (total_bytes / (1024.0 ** 3)) / total_seconds
        if total_seconds > 0
        else None,
    }


def real(args: argparse.Namespace) -> dict[str, Any]:
    if not hasattr(os, "O_DIRECT"):
        raise RuntimeError("real queue-depth probe requires Linux O_DIRECT")
    if not hasattr(os, "preadv"):
        raise RuntimeError("real queue-depth probe requires os.preadv")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    trunk = work_dir / "decoder64.k3.bin"
    manifest_path = work_dir / "decoder64.k3.json"

    directory = parse_gguf(Path(args.model))
    manifest = pack_gguf_layers(
        directory,
        trunk,
        manifest_path,
        layers=range(N_LAYER),
        model_id=MODEL_ID,
        revision=REVISION,
        source_sha256=GGUF_SHA256,
        expected_layers=N_LAYER,
    )
    layer_meta = {int(row["layer"]): row for row in manifest["layers"]}
    total_k3 = sum(int(row["read_bytes"]) for row in manifest["layers"])
    if total_k3 != EXPECTED_K3_BYTES:
        raise RuntimeError(f"K3 bytes {total_k3} != {EXPECTED_K3_BYTES}")

    max_layer = max(int(row["read_bytes"]) for row in manifest["layers"])
    reader = K3Trunk(
        trunk,
        manifest_path,
        budget_bytes=2 * max_layer,
        want_ring=2,
        max_pinned=0,
        prefer_direct_io=True,
    )
    try:
        if not reader.direct_io:
            raise RuntimeError("queue-depth probe requires direct I/O; buffered fallback is invalid")
        if reader.plan.ring_slots != 2:
            raise RuntimeError(f"expected 2 ring slots, got {reader.plan.ring_slots}")
        if reader.plan.planned_bytes != 2 * reader.plan.slot_bytes:
            raise RuntimeError("ring memory plan changed unexpectedly")
        target = reader._ring[0]  # measurement-only: write one existing slot, keep residency unchanged.

        methods: list[tuple[int | None, int | None]] = [(None, None)]
        methods.extend((block, qd) for block in BLOCK_SIZES for qd in QUEUE_DEPTHS)

        # Two rounds with reversed method order reduce systematic first/last-method drift.
        expected_sha: dict[int, str] = {}
        records: list[dict[str, Any]] = []
        for round_index, order in enumerate((methods, list(reversed(methods)))):
            for block_bytes, qd in order:
                rec = _run_method(
                    fd=reader.fd,
                    target=target,
                    layer_meta=layer_meta,
                    layers=SAMPLE_LAYERS,
                    block_bytes=block_bytes,
                    queue_depth=qd,
                    expected_sha=expected_sha,
                )
                rec["round"] = round_index
                records.append(rec)

        aggregate: dict[str, dict[str, Any]] = {}
        for rec in records:
            name = str(rec["name"])
            row = aggregate.setdefault(
                name,
                {
                    "name": name,
                    "block_bytes": rec["block_bytes"],
                    "queue_depth": rec["queue_depth"],
                    "rounds": 0,
                    "read_bytes": 0,
                    "seconds": 0.0,
                    "preadv_calls": 0,
                },
            )
            row["rounds"] += 1
            row["read_bytes"] += int(rec["read_bytes"])
            row["seconds"] += float(rec["seconds"])
            row["preadv_calls"] += int(rec["preadv_calls"])
        for row in aggregate.values():
            row["gib_s"] = (
                (row["read_bytes"] / (1024.0 ** 3)) / row["seconds"]
                if row["seconds"] > 0
                else None
            )

        baseline = aggregate["one_shot_current"]
        for row in aggregate.values():
            row["speedup_vs_current_one_shot"] = (
                baseline["seconds"] / row["seconds"] if row["seconds"] > 0 else None
            )

        candidates = [row for name, row in aggregate.items() if name != "one_shot_current"]
        best = min(candidates, key=lambda row: float(row["seconds"]))
        sample_bytes = sum(int(layer_meta[layer]["read_bytes"]) for layer in SAMPLE_LAYERS)

        report = {
            "status": "PASS",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "gguf_sha256": GGUF_SHA256,
            "expected_k3_bytes": EXPECTED_K3_BYTES,
            "actual_k3_bytes": total_k3,
            "sample_layers": list(SAMPLE_LAYERS),
            "sample_bytes_per_method_round": sample_bytes,
            "round_count": 2,
            "ring": {
                "direct_io": bool(reader.direct_io),
                "ring_slots": reader.plan.ring_slots,
                "slot_bytes": reader.plan.slot_bytes,
                "planned_bytes": reader.plan.planned_bytes,
                "budget_bytes": reader.plan.budget_bytes,
            },
            "records": records,
            "aggregate": aggregate,
            "best_candidate": best,
            "performance_positive": float(best["seconds"]) < float(baseline["seconds"]),
            "reference_sha256_by_layer": {
                str(layer): digest for layer, digest in sorted(expected_sha.items())
            },
            "max_rss_gib": rss_gib(),
        }
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "status": report["status"],
            "baseline": baseline,
            "best_candidate": best,
            "ring": report["ring"],
            "sample_bytes_per_method_round": sample_bytes,
            "max_rss_gib": report["max_rss_gib"],
        }, indent=2, sort_keys=True))
        print("QWEN38_K3_IO_QD_REAL_PASS")
        return report
    finally:
        reader.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    sub.add_parser("synthetic")
    real_p = sub.add_parser("real")
    real_p.add_argument("--model", required=True, type=Path)
    real_p.add_argument("--work-dir", required=True, type=Path)
    real_p.add_argument("--output", required=True, type=Path)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "sanity":
        sanity()
    elif args.cmd == "synthetic":
        synthetic()
    else:
        real(args)


if __name__ == "__main__":
    main()
