#!/usr/bin/env python3
"""Win32 integration gate: progressive native ring -> exact quant/F32 kernels.

The test is deliberately small and model-free.  It places valid Q6_K, Q8_0,
and F32 matrices inside a two-layer progressive K3 fixture, reads them through
the native NO_BUFFERING|OVERLAPPED two-slot ring, then feeds the resulting
writable memoryviews directly into the same ctypes runtimes used by the decoder.

Arithmetic is checked against the identical native kernels fed from ordinary
bytearray-backed memoryviews.  The only variable is the weight-buffer origin.
"""
from __future__ import annotations

from array import array
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Sequence

from qwen38_win32_bootstrap import install_resource_compat

install_resource_compat()

import qwen35_gdn_quant_layer_gate as gdn
from k3_stream import ALIGN
from native_f32_runtime import NativeF32Runtime, load_f32_lib
from q6_persistent_pool_runtime import Q6PersistentPoolRuntime
from qwen38_k3_progressive import LAYOUT_POLICY
from qwen38_k3_progressive_win32 import Win32ProgressiveK3Trunk


N = 256
ROWS = 8
VECTORS = 2
F32_N = 64
Q6_BLOCK_BYTES = 210
Q8_BLOCK_BYTES = 34
LAYER_BYTES = 3 * ALIGN


def _f16_scale() -> bytes:
    return b"\x00\x28"


def _q6_weights(layer: int) -> bytes:
    out = bytearray()
    for row in range(ROWS):
        for block_index in range(N // 256):
            block = bytearray(Q6_BLOCK_BYTES)
            for i in range(128):
                block[i] = (17 * i + 11 * row + 23 * layer + block_index) & 0xFF
            for i in range(64):
                block[128 + i] = (29 * i + 7 * row + 31 * layer + block_index) & 0xFF
            for i in range(16):
                scale = ((i + 3 * row + 5 * layer) % 15) + 1
                block[192 + i] = scale & 0xFF
            block[208:210] = _f16_scale()
            out.extend(block)
    expected = ROWS * (N // 256) * Q6_BLOCK_BYTES
    if len(out) != expected:
        raise AssertionError((len(out), expected))
    return bytes(out)


def _q8_weights(layer: int) -> bytes:
    out = bytearray()
    for row in range(ROWS):
        for block_index in range(N // 32):
            block = bytearray(Q8_BLOCK_BYTES)
            block[0:2] = _f16_scale()
            for i in range(32):
                value = ((i * 5 + row * 3 + block_index * 7 + layer * 11) % 31) - 15
                block[2 + i] = value & 0xFF
            out.extend(block)
    expected = ROWS * (N // 32) * Q8_BLOCK_BYTES
    if len(out) != expected:
        raise AssertionError((len(out), expected))
    return bytes(out)


def _f32_weights(layer: int) -> bytes:
    values = [
        ((row * 37 + col * 13 + layer * 19) % 101 - 50) / 37.0
        for row in range(ROWS)
        for col in range(F32_N)
    ]
    return struct.pack("<" + "f" * len(values), *values)


def _build_fixture(root: Path):
    trunk = root / "win32-ring-kernel-fixture.bin"
    index = root / "win32-ring-kernel-fixture.json"
    full = bytearray(2 * LAYER_BYTES)
    expected: dict[str, bytes] = {}
    layers = []
    tensor_index = {}

    for layer in range(2):
        entries = (
            (f"blk.{layer}.fixture_q6.weight", 0, _q6_weights(layer), "Q6_K", [N, ROWS]),
            (f"blk.{layer}.fixture_q8.weight", ALIGN, _q8_weights(layer), "Q8_0", [N, ROWS]),
            (f"blk.{layer}.fixture_f32.weight", 2 * ALIGN, _f32_weights(layer), "F32", [F32_N, ROWS]),
        )
        tensors = []
        base = layer * LAYER_BYTES
        for name, offset, payload, type_name, shape in entries:
            if len(payload) > ALIGN:
                raise RuntimeError(f"{name}: fixture tensor exceeds one frontier page")
            full[base + offset:base + offset + len(payload)] = payload
            expected[name] = payload
            meta = {
                "name": name,
                "layer": layer,
                "offset": offset,
                "nbytes": len(payload),
                "type_name": type_name,
                "shape": shape,
            }
            tensors.append(meta)
            tensor_index[name] = {
                "layer": layer,
                "offset": offset,
                "nbytes": len(payload),
            }

        layers.append({
            "layer": layer,
            "kind": "synthetic-kernel-integration",
            "file_offset": base,
            "data_bytes": 2 * ALIGN + len(entries[-1][2]),
            "read_bytes": LAYER_BYTES,
            "tensor_count": len(tensors),
            "tensors": tensors,
            "readiness_frontiers": [
                {"stage": "q6", "ready_bytes": ALIGN, "ready_fraction": 1 / 3},
                {"stage": "q8", "ready_bytes": 2 * ALIGN, "ready_fraction": 2 / 3},
                {"stage": "f32", "ready_bytes": 3 * ALIGN, "ready_fraction": 1.0},
            ],
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


def _rows_f32_bytes(rows: Sequence[Sequence[float]]) -> bytes:
    flat = array("f", (float(value) for row in rows for value in row))
    return flat.tobytes()


def _vector_f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def _await_complete(reader: Win32ProgressiveK3Trunk, bound) -> None:
    with reader._cv:
        state = reader._states[bound.slot]
        while state.layer == bound.layer and state.status == "loading":
            reader._cv.wait()
            state = reader._states[bound.slot]
        if state.status != "complete" or state.ready_bytes != state.read_bytes:
            raise RuntimeError(f"layer {bound.layer}: incomplete state {state}")


def run(args) -> dict:
    if os.name != "nt":
        raise RuntimeError("Win32 ring/kernel integration requires Windows")

    xs = [
        [((i * 17 + v * 23) % 97 - 48) / 19.0 for i in range(N)]
        for v in range(VECTORS)
    ]
    f32_x = [((i * 11) % 67 - 33) / 17.0 for i in range(F32_N)]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        trunk, index, expected = _build_fixture(root)
        budget = 2 * LAYER_BYTES
        reader = Win32ProgressiveK3Trunk(
            trunk,
            index,
            budget_bytes=budget,
            win32_direct_lib=args.win32_direct_lib,
            want_ring=2,
            max_pinned=0,
        )
        base = gdn.QuantRuntime(gdn._load_native(args.quant_lib))
        quant = Q6PersistentPoolRuntime(
            base,
            args.q6_pool_lib,
            threads=2,
            max_rows=ROWS,
            max_vec=VECTORS,
        )
        f32 = NativeF32Runtime(base, load_f32_lib(args.f32_lib))
        checks = []
        try:
            for layer in range(2):
                bound = reader.bind(layer)
                if layer == 0 and not reader.prefetch(1):
                    raise RuntimeError("prefetch(1) was not accepted")

                q6_name = f"blk.{layer}.fixture_q6.weight"
                q6_view = reader.tensor_view(bound, q6_name)
                if hashlib.sha256(q6_view).digest() != hashlib.sha256(expected[q6_name]).digest():
                    raise RuntimeError(f"{q6_name}: ring payload mismatch")
                q6_copy = memoryview(bytearray(expected[q6_name]))
                q6_meta = {"name": q6_name, "type_name": "Q6_K", "shape": [N, ROWS]}
                q6_ref = quant.matvec_many(q6_copy, q6_meta, xs)
                q6_ring = quant.matvec_many(q6_view, q6_meta, xs)
                if _rows_f32_bytes(q6_ref) != _rows_f32_bytes(q6_ring):
                    raise RuntimeError(f"{q6_name}: ring/copy Q6 bitwise mismatch")
                checks.append({"layer": layer, "kind": "Q6_K", "bitwise_exact": True})
                q6_copy.release()
                q6_view.release()

                q8_name = f"blk.{layer}.fixture_q8.weight"
                q8_view = reader.tensor_view(bound, q8_name)
                if hashlib.sha256(q8_view).digest() != hashlib.sha256(expected[q8_name]).digest():
                    raise RuntimeError(f"{q8_name}: ring payload mismatch")
                q8_copy = memoryview(bytearray(expected[q8_name]))
                q8_meta = {"name": q8_name, "type_name": "Q8_0", "shape": [N, ROWS]}
                q8_ref = quant.matvec_many(q8_copy, q8_meta, xs)
                q8_ring = quant.matvec_many(q8_view, q8_meta, xs)
                if _rows_f32_bytes(q8_ref) != _rows_f32_bytes(q8_ring):
                    raise RuntimeError(f"{q8_name}: ring/copy Q8 bitwise mismatch")
                checks.append({"layer": layer, "kind": "Q8_0", "bitwise_exact": True})
                q8_copy.release()
                q8_view.release()

                f32_name = f"blk.{layer}.fixture_f32.weight"
                f32_view = reader.tensor_view(bound, f32_name)
                if hashlib.sha256(f32_view).digest() != hashlib.sha256(expected[f32_name]).digest():
                    raise RuntimeError(f"{f32_name}: ring payload mismatch")
                f32_copy = memoryview(bytearray(expected[f32_name]))
                f32_meta = {"name": f32_name, "type_name": "F32", "shape": [F32_N, ROWS]}
                f32_ref = f32.matvec(f32_copy, f32_meta, f32_x)
                f32_ring = f32.matvec(f32_view, f32_meta, f32_x)
                if _vector_f32_bytes(f32_ref) != _vector_f32_bytes(f32_ring):
                    raise RuntimeError(f"{f32_name}: ring/copy F32 bitwise mismatch")
                checks.append({"layer": layer, "kind": "F32", "bitwise_exact": True})
                f32_copy.release()
                f32_view.release()

                _await_complete(reader, bound)
                bound.release()

            reader_report = reader.report()
            pool_report = quant.pool_report()
            many_report = quant.report()
            f32_report = f32.report()
            expected_read = 2 * LAYER_BYTES
            if reader_report["bytes_read"] != expected_read:
                raise RuntimeError(f"reader byte contract changed: {reader_report}")
            if reader_report["win32_native_read_bytes"] != expected_read:
                raise RuntimeError(f"native read byte contract changed: {reader_report}")
            if reader_report["pread_calls"] != 6 or reader_report["ready_events"] != 6:
                raise RuntimeError(f"frontier contract changed: {reader_report}")
            if reader_report["ring_slots"] != 2 or reader_report["native_ring_slots"] != 2:
                raise RuntimeError(f"ring contract changed: {reader_report}")
            if reader_report["storage_io_concurrency"] != 1:
                raise RuntimeError(f"storage concurrency changed: {reader_report}")
            if reader_report["max_queued_requests_observed"] > 1:
                raise RuntimeError(f"deferred queue depth changed: {reader_report}")
            if not reader_report["win32_no_buffering"] or not reader_report["win32_overlapped"]:
                raise RuntimeError(f"Win32 direct-I/O flags missing: {reader_report}")
            if pool_report["threads"] != 2 or pool_report["calls"] != 4:
                raise RuntimeError(f"Q6 pool coverage changed: {pool_report}")
            if pool_report["native_calls"] != 4:
                raise RuntimeError(f"Q6 native coverage changed: {pool_report}")
            if many_report["many_calls"] != 8:
                raise RuntimeError(f"quant-many coverage changed: {many_report}")
            if f32_report["native_f32_calls"] != 4:
                raise RuntimeError(f"F32 coverage changed: {f32_report}")

            payload = {
                "schema": "qwen38-win32-ring-kernel-integration-v1",
                "status": "PASS",
                "checks": checks,
                "check_count": len(checks),
                "reader": reader_report,
                "q6_pool": pool_report,
                "quant_many": many_report,
                "native_f32": f32_report,
                "ring_weight_views_direct": True,
                "model_loaded": False,
                "arithmetic_change": False,
                "layout_change": False,
                "scheduler_change": False,
                "linux_runtime_changed": False,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            print("QWEN38_WIN32_RING_KERNEL_BITWISE_PASS")
            return payload
        finally:
            quant.close()
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--win32-direct-lib", type=Path, required=True)
    parser.add_argument("--quant-lib", type=Path, required=True)
    parser.add_argument("--q6-pool-lib", type=Path, required=True)
    parser.add_argument("--f32-lib", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
