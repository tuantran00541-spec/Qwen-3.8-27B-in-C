#!/usr/bin/env python3
"""Q6_K-only persistent row-pool overlay for the exact Qwen3.8 many runtime.

Q8_0 delegates unchanged to FastMarshalQuantManyRuntime.  Q6_K uses the same
activation preparation, contiguous marshaling, output layout, and proven exact
many-row arithmetic; only independent output-row ranges are dispatched to a
persistent native worker pool.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import time
from typing import Any, Sequence

from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime, marshal_output_bulk
from quant_many_runtime import load_many_lib


def load_q6_pool_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.qwen_q6_pool_create.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    lib.qwen_q6_pool_create.restype = ctypes.c_void_p
    lib.qwen_q6_pool_destroy.argtypes = [ctypes.c_void_p]
    lib.qwen_q6_pool_destroy.restype = None
    lib.qwen_q6_pool_matvec_many.argtypes = [
        ctypes.c_void_p,
        u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        u8p, ctypes.c_size_t, ctypes.c_size_t, fp,
    ]
    lib.qwen_q6_pool_matvec_many.restype = ctypes.c_int
    lib.qwen_q6_pool_calls.argtypes = [ctypes.c_void_p]
    lib.qwen_q6_pool_calls.restype = ctypes.c_uint64
    lib.qwen_q6_pool_threads.argtypes = [ctypes.c_void_p]
    lib.qwen_q6_pool_threads.restype = ctypes.c_int
    return lib


class Q6PersistentPoolRuntime(FastMarshalQuantManyRuntime):
    def __init__(
        self,
        runtime,
        library: Path,
        *,
        threads: int,
        max_rows: int = 17408,
        max_vec: int = 11,
    ):
        threads = int(threads)
        max_rows = int(max_rows)
        max_vec = int(max_vec)
        if threads < 1:
            raise ValueError("Q6 pool threads must be positive")
        if max_rows < 1 or max_vec < 1:
            raise ValueError("Q6 pool capacities must be positive")
        super().__init__(runtime, load_many_lib(library))
        self.pool_lib = load_q6_pool_lib(library)
        self.threads = threads
        self.max_rows = max_rows
        self.max_vec = max_vec
        self.handle = self.pool_lib.qwen_q6_pool_create(threads, max_rows, max_vec)
        if not self.handle:
            raise RuntimeError(
                f"could not create persistent Q6 pool threads={threads} "
                f"max_rows={max_rows} max_vec={max_vec}")
        self.q6_pool_calls = 0
        self.q6_pool_vectors = 0
        self.q6_pool_rows = 0
        self.q6_pool_seconds = 0.0

    def close(self) -> None:
        if self.handle:
            self.pool_lib.qwen_q6_pool_destroy(self.handle)
            self.handle = None

    def matvec_many(
        self,
        weights: memoryview,
        meta: dict[str, Any],
        xs: Sequence[Sequence[float]],
        prepared=None,
    ) -> list[list[float]]:
        kind = str(meta["type_name"])
        if kind != "Q6_K":
            return super().matvec_many(weights, meta, xs, prepared=prepared)
        if not self.handle:
            raise RuntimeError("persistent Q6 pool is closed")

        rows_in = [list(map(float, x)) for x in xs]
        if not rows_in:
            return []
        ne0, rows = map(int, meta["shape"])
        if rows > self.max_rows:
            raise ValueError(f"{meta['name']}: Q6 rows={rows} exceed pool max_rows={self.max_rows}")
        if len(rows_in) > self.max_vec:
            raise ValueError(
                f"{meta['name']}: vectors={len(rows_in)} exceed pool max_vec={self.max_vec}")
        if any(len(x) != ne0 for x in rows_in):
            raise ValueError(f"{meta['name']}: matvec_many input width != {ne0}")
        if prepared is None:
            prepared = self.prepare_many(rows_in, kind)
        if len(prepared) != len(rows_in):
            raise ValueError(f"{meta['name']}: prepared vector count mismatch")
        act_bytes = int(prepared[0][1])
        if any(int(n) != act_bytes for _, n in prepared):
            raise ValueError(f"{meta['name']}: activation byte sizes differ")

        blob = b"".join(bytes(buf) for buf, _ in prepared)
        acts = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
        out = (ctypes.c_float * (len(rows_in) * rows))()

        t0 = time.monotonic()
        rc = self.pool_lib.qwen_q6_pool_matvec_many(
            self.handle,
            w_arr, len(weights), rows, ne0,
            acts, act_bytes, len(rows_in), out,
        )
        self.q6_pool_seconds += time.monotonic() - t0
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: persistent Q6 pool rc={rc}")

        self.many_calls += 1
        self.many_vectors += len(rows_in)
        self.many_rows += len(rows_in) * rows
        self.q6_pool_calls += 1
        self.q6_pool_vectors += len(rows_in)
        self.q6_pool_rows += len(rows_in) * rows
        return marshal_output_bulk(out, rows, len(rows_in))

    def pool_report(self) -> dict[str, float | int]:
        native_calls = 0 if not self.handle else int(self.pool_lib.qwen_q6_pool_calls(self.handle))
        native_threads = self.threads if not self.handle else int(
            self.pool_lib.qwen_q6_pool_threads(self.handle))
        return {
            "threads": native_threads,
            "calls": self.q6_pool_calls,
            "native_calls": native_calls,
            "vectors": self.q6_pool_vectors,
            "rows": self.q6_pool_rows,
            "seconds": self.q6_pool_seconds,
            "max_rows": self.max_rows,
            "max_vec": self.max_vec,
        }
