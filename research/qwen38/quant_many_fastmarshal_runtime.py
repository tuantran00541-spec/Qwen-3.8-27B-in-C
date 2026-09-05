#!/usr/bin/env python3
"""Exact QuantManyRuntime variant with bulk F32 output marshaling only.

C matvec arithmetic, activation preparation, input copying, and output layout are
unchanged. The only difference from QuantManyRuntime is converting the flat
ctypes c_float output buffer to Python rows in one bulk byte copy instead of
calling float(...) once per output element.
"""
from __future__ import annotations

from array import array
import ctypes
from typing import Any, Sequence

from quant_many_runtime import QuantManyRuntime


def marshal_output_bulk(out, rows: int, n_vec: int) -> list[list[float]]:
    if rows < 0 or n_vec < 0:
        raise ValueError("negative output shape")
    total = rows * n_vec
    if total == 0:
        return []
    if len(out) != total:
        raise ValueError(f"output length {len(out)} != {total}")

    flat = array("f")
    flat.frombytes(memoryview(out).cast("B"))
    if len(flat) != total:
        raise RuntimeError("bulk F32 marshal length mismatch")
    return [flat[v * rows:(v + 1) * rows].tolist() for v in range(n_vec)]


class FastMarshalQuantManyRuntime(QuantManyRuntime):
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
        return marshal_output_bulk(out, rows, len(rows_in))
