#!/usr/bin/env python3
"""Exact multi-vector quantized matvec wrapper for Qwen3.8 prefill research.

The wrapped scalar runtime remains authoritative for F32 and single-vector
calls.  Q6_K/Q8_0 ``matvec_many`` uses the real-model-validated bridge that
preserves each vector's original accumulation order while amortizing one weight
row traversal/unpack across multiple prompt activations.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Sequence


def load_many_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    args = [
        u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        u8p, ctypes.c_size_t, ctypes.c_size_t, fp,
    ]
    for name in (
        "qwen_matvec_many_q8_0_q8_0_bridge",
        "qwen_matvec_many_q6_k_q8_k_bridge",
    ):
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int
    return lib


class QuantManyRuntime:
    def __init__(self, runtime, many_lib):
        self.runtime = runtime
        self.many_lib = many_lib
        self.many_calls = 0
        self.many_vectors = 0
        self.many_rows = 0

    def quantize(self, x: Sequence[float], kind: str):
        return self.runtime.quantize(x, kind)

    def matvec(self, weights, meta, x, prepared=None):
        return self.runtime.matvec(weights, meta, x, prepared=prepared)

    @property
    def activation_quantizations(self):
        return self.runtime.activation_quantizations

    @property
    def matvec_rows(self):
        return self.runtime.matvec_rows

    def prepare_many(self, xs: Sequence[Sequence[float]], kind: str):
        return [self.runtime.quantize(x, kind) for x in xs]

    def matvec_many(
        self,
        weights: memoryview,
        meta: dict[str, Any],
        xs: Sequence[Sequence[float]],
        prepared=None,
    ) -> list[list[float]]:
        rows_in = [list(map(float, x)) for x in xs]
        if not rows_in:
            return []
        kind = str(meta["type_name"])
        if kind not in {"Q6_K", "Q8_0"}:
            return [self.runtime.matvec(weights, meta, x) for x in rows_in]

        ne0, rows = map(int, meta["shape"])
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
        fn = (
            self.many_lib.qwen_matvec_many_q6_k_q8_k_bridge
            if kind == "Q6_K"
            else self.many_lib.qwen_matvec_many_q8_0_q8_0_bridge
        )
        rc = fn(
            w_arr, len(weights), rows, ne0,
            acts, act_bytes, len(rows_in), out,
        )
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: exact matvec_many rc={rc}")

        self.many_calls += 1
        self.many_vectors += len(rows_in)
        self.many_rows += len(rows_in) * rows
        return [
            [float(out[v * rows + r]) for r in range(rows)]
            for v in range(len(rows_in))
        ]

    def report(self) -> dict[str, int]:
        return {
            "many_calls": self.many_calls,
            "many_vectors": self.many_vectors,
            "many_rows": self.many_rows,
        }


def enable_quant_many(engine, many_lib_path: Path) -> QuantManyRuntime:
    if isinstance(engine.runtime, QuantManyRuntime):
        return engine.runtime
    wrapped = QuantManyRuntime(engine.runtime, load_many_lib(many_lib_path))
    engine.runtime = wrapped
    return wrapped
