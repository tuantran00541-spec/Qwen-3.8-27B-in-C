#!/usr/bin/env python3
"""Warm, interleaved K3 Direct-I/O A/B for chunking vs queue depth.

The first queue-depth sweep showed a large first-read effect: the current
one-shot baseline was slow in round 0 but matched chunked reads in round 1.
This follow-up removes that confound by:
  * warming every sampled layer before timing;
  * measuring five methods across five cyclic orders so every method occupies
    every order position exactly once;
  * preserving the exact two-slot K3 ring budget and Direct-I/O path;
  * hash-checking every read against the warmup bytes.

This is measurement-only.  It does not change the production reader/runtime.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import time
from typing import Any

from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk
import qwen38_k3_io_queue_depth_probe as qd

METHODS: tuple[tuple[str, int | None, int | None], ...] = (
    ("one_shot_current", None, None),
    ("chunk_1m_qd1", 1 * 1024 * 1024, 1),
    ("chunk_1m_qd4", 1 * 1024 * 1024, 4),
    ("chunk_4m_qd1", 4 * 1024 * 1024, 1),
    ("chunk_4m_qd4", 4 * 1024 * 1024, 4),
)
ROUNDS = len(METHODS)
SAMPLE_LAYERS = qd.SAMPLE_LAYERS


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def sanity() -> None:
    qd.sanity()
    names = [m[0] for m in METHODS]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate interleaved method name")
    if ROUNDS != len(METHODS):
        raise SystemExit("cyclic design requires one round per method")
    seen_positions = {name: set() for name in names}
    for round_index in range(ROUNDS):
        order = METHODS[round_index:] + METHODS[:round_index]
        for position, method in enumerate(order):
            seen_positions[method[0]].add(position)
    if any(pos != set(range(ROUNDS)) for pos in seen_positions.values()):
        raise SystemExit("cyclic order is not position-balanced")
    print("QWEN38_K3_IO_INTERLEAVED_SANITY PASS")


def _read_method(
    *,
    fd: int,
    target,
    nbytes: int,
    file_offset: int,
    block_bytes: int | None,
    queue_depth: int | None,
    executor: ThreadPoolExecutor | None,
) -> tuple[int, int]:
    if block_bytes is None:
        return qd._read_one_shot(fd, target, nbytes, file_offset)
    return qd._read_chunked(
        fd,
        target,
        nbytes,
        file_offset,
        int(block_bytes),
        int(queue_depth),
        executor,
    )


def real(args: argparse.Namespace) -> dict[str, Any]:
    sanity()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    trunk = work_dir / "decoder64.k3.bin"
    manifest_path = work_dir / "decoder64.k3.json"

    directory = parse_gguf(Path(args.model))
    manifest = pack_gguf_layers(
        directory,
        trunk,
        manifest_path,
        layers=range(qd.N_LAYER),
        model_id=qd.MODEL_ID,
        revision=qd.REVISION,
        source_sha256=qd.GGUF_SHA256,
        expected_layers=qd.N_LAYER,
    )
    layer_meta = {int(row["layer"]): row for row in manifest["layers"]}
    total_k3 = sum(int(row["read_bytes"]) for row in manifest["layers"])
    if total_k3 != qd.EXPECTED_K3_BYTES:
        raise RuntimeError(f"K3 bytes {total_k3} != {qd.EXPECTED_K3_BYTES}")

    max_layer = max(int(row["read_bytes"]) for row in manifest["layers"])
    reader = K3Trunk(
        trunk,
        manifest_path,
        budget_bytes=2 * max_layer,
        want_ring=2,
        max_pinned=0,
        prefer_direct_io=True,
    )
    executors: dict[str, ThreadPoolExecutor] = {}
    try:
        if not reader.direct_io:
            raise RuntimeError("interleaved probe requires Direct I/O")
        if reader.plan.ring_slots != 2:
            raise RuntimeError(f"expected 2 ring slots, got {reader.plan.ring_slots}")
        if reader.plan.planned_bytes != 672_899_072:
            raise RuntimeError(
                f"ring planned bytes changed: {reader.plan.planned_bytes} != 672899072"
            )
        target = reader._ring[0]

        for name, block_bytes, queue_depth in METHODS:
            if block_bytes is not None and int(queue_depth) > 1:
                executors[name] = ThreadPoolExecutor(max_workers=int(queue_depth))

        expected_sha: dict[int, str] = {}
        warmup_seconds = 0.0
        warmup_bytes = 0
        for layer in SAMPLE_LAYERS:
            meta = layer_meta[layer]
            nbytes = int(meta["read_bytes"])
            offset = int(meta["file_offset"])
            t0 = time.monotonic()
            got, _ = qd._read_one_shot(reader.fd, target, nbytes, offset)
            warmup_seconds += time.monotonic() - t0
            if got != nbytes:
                raise RuntimeError(f"warmup layer {layer}: {got} != {nbytes}")
            expected_sha[layer] = qd._sha256_view(target, nbytes)
            warmup_bytes += got

        records: list[dict[str, Any]] = []
        for round_index in range(ROUNDS):
            order = METHODS[round_index:] + METHODS[:round_index]
            layer_order = SAMPLE_LAYERS if round_index % 2 == 0 else tuple(reversed(SAMPLE_LAYERS))
            for position, (name, block_bytes, queue_depth) in enumerate(order):
                total_seconds = 0.0
                total_bytes = 0
                total_calls = 0
                layer_rows: list[dict[str, Any]] = []
                executor = executors.get(name)
                for layer in layer_order:
                    meta = layer_meta[layer]
                    nbytes = int(meta["read_bytes"])
                    offset = int(meta["file_offset"])
                    t0 = time.monotonic()
                    got, calls = _read_method(
                        fd=reader.fd,
                        target=target,
                        nbytes=nbytes,
                        file_offset=offset,
                        block_bytes=block_bytes,
                        queue_depth=queue_depth,
                        executor=executor,
                    )
                    seconds = time.monotonic() - t0
                    if got != nbytes:
                        raise RuntimeError(f"{name} layer {layer}: {got} != {nbytes}")
                    digest = qd._sha256_view(target, nbytes)
                    if digest != expected_sha[layer]:
                        raise RuntimeError(
                            f"{name} layer {layer}: byte mismatch {digest} != {expected_sha[layer]}"
                        )
                    total_seconds += seconds
                    total_bytes += got
                    total_calls += calls
                    layer_rows.append({
                        "layer": layer,
                        "seconds": seconds,
                        "read_bytes": got,
                        "preadv_calls": calls,
                        "gib_s": (got / (1024.0 ** 3)) / seconds if seconds > 0 else None,
                    })
                records.append({
                    "round": round_index,
                    "position": position,
                    "name": name,
                    "block_bytes": block_bytes,
                    "queue_depth": queue_depth,
                    "seconds": total_seconds,
                    "read_bytes": total_bytes,
                    "preadv_calls": total_calls,
                    "gib_s": (total_bytes / (1024.0 ** 3)) / total_seconds
                    if total_seconds > 0 else None,
                    "layers": layer_rows,
                })

        aggregate: dict[str, dict[str, Any]] = {}
        for rec in records:
            name = str(rec["name"])
            row = aggregate.setdefault(name, {
                "name": name,
                "block_bytes": rec["block_bytes"],
                "queue_depth": rec["queue_depth"],
                "rounds": 0,
                "read_bytes": 0,
                "seconds": 0.0,
                "preadv_calls": 0,
                "positions": [],
            })
            row["rounds"] += 1
            row["read_bytes"] += int(rec["read_bytes"])
            row["seconds"] += float(rec["seconds"])
            row["preadv_calls"] += int(rec["preadv_calls"])
            row["positions"].append(int(rec["position"]))

        for row in aggregate.values():
            row["gib_s"] = (
                (row["read_bytes"] / (1024.0 ** 3)) / row["seconds"]
                if row["seconds"] > 0 else None
            )

        baseline = aggregate["one_shot_current"]
        for row in aggregate.values():
            row["speedup_vs_current_one_shot"] = (
                baseline["seconds"] / row["seconds"] if row["seconds"] > 0 else None
            )

        q1_1m = aggregate["chunk_1m_qd1"]
        q4_1m = aggregate["chunk_1m_qd4"]
        q1_4m = aggregate["chunk_4m_qd1"]
        q4_4m = aggregate["chunk_4m_qd4"]
        pairwise = {
            "chunk_1m_qd1_vs_current": baseline["seconds"] / q1_1m["seconds"],
            "chunk_4m_qd1_vs_current": baseline["seconds"] / q1_4m["seconds"],
            "qd4_vs_qd1_at_1m": q1_1m["seconds"] / q4_1m["seconds"],
            "qd4_vs_qd1_at_4m": q1_4m["seconds"] / q4_4m["seconds"],
            "one_mib_vs_four_mib_at_qd1": q1_4m["seconds"] / q1_1m["seconds"],
            "one_mib_vs_four_mib_at_qd4": q4_4m["seconds"] / q4_1m["seconds"],
        }
        best = min(aggregate.values(), key=lambda row: float(row["seconds"]))
        report = {
            "status": "PASS",
            "model_id": qd.MODEL_ID,
            "revision": qd.REVISION,
            "gguf_sha256": qd.GGUF_SHA256,
            "actual_k3_bytes": total_k3,
            "sample_layers": list(SAMPLE_LAYERS),
            "sample_bytes_per_method_round": warmup_bytes,
            "warmup": {
                "read_bytes": warmup_bytes,
                "seconds": warmup_seconds,
                "gib_s": (warmup_bytes / (1024.0 ** 3)) / warmup_seconds
                if warmup_seconds > 0 else None,
            },
            "round_count": ROUNDS,
            "ring": {
                "direct_io": bool(reader.direct_io),
                "ring_slots": reader.plan.ring_slots,
                "slot_bytes": reader.plan.slot_bytes,
                "planned_bytes": reader.plan.planned_bytes,
                "budget_bytes": reader.plan.budget_bytes,
            },
            "records": records,
            "aggregate": aggregate,
            "pairwise": pairwise,
            "best_method": best,
            "queue_depth_material_ge_2pct": (
                pairwise["qd4_vs_qd1_at_1m"] >= 1.02
                or pairwise["qd4_vs_qd1_at_4m"] >= 1.02
            ),
            "chunking_material_ge_2pct": (
                pairwise["chunk_1m_qd1_vs_current"] >= 1.02
                or pairwise["chunk_4m_qd1_vs_current"] >= 1.02
            ),
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
            "warmup": report["warmup"],
            "aggregate": report["aggregate"],
            "pairwise": report["pairwise"],
            "best_method": report["best_method"],
            "queue_depth_material_ge_2pct": report["queue_depth_material_ge_2pct"],
            "chunking_material_ge_2pct": report["chunking_material_ge_2pct"],
            "ring": report["ring"],
            "max_rss_gib": report["max_rss_gib"],
        }, indent=2, sort_keys=True))
        print("QWEN38_K3_IO_INTERLEAVED_REAL_PASS")
        return report
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)
        reader.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    real_p = sub.add_parser("real")
    real_p.add_argument("--model", required=True, type=Path)
    real_p.add_argument("--work-dir", required=True, type=Path)
    real_p.add_argument("--output", required=True, type=Path)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        real(args)


if __name__ == "__main__":
    main()
