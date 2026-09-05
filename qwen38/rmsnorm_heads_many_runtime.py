#!/usr/bin/env python3
"""ctypes bridge for exact sequential many-token Q/K head RMSNorm."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


def _ptr(buf: array):
    return (ctypes.c_float * len(buf)).from_buffer(buf)


class ExactRMSNormHeadsMany:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        fp = ctypes.POINTER(ctypes.c_float)
        self.lib.qwen38_rmsnorm_heads_many_exact_f32.argtypes = [
            fp,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            fp,
            ctypes.c_float,
            fp,
        ]
        self.lib.qwen38_rmsnorm_heads_many_exact_f32.restype = ctypes.c_int
        self.calls = 0
        self.token_rows = 0
        self.head_rows = 0
        self.values = 0

    def compute(
        self,
        rows: Sequence[Sequence[float]],
        heads: int,
        weight: Sequence[float],
        eps: float,
    ) -> list[list[float]]:
        n_rows = len(rows)
        if n_rows == 0:
            return []
        heads = int(heads)
        head_dim = len(weight)
        if heads <= 0 or head_dim <= 0:
            raise ValueError("invalid head RMSNorm-many shape")
        width = heads * head_dim
        values = array("f")
        for row in rows:
            if len(row) != width:
                raise ValueError(
                    f"head RMSNorm-many row width={len(row)} expected={width}")
            values.extend(row)
        w = array("f", weight)
        out = array("f", [0.0]) * (n_rows * width)
        rc = self.lib.qwen38_rmsnorm_heads_many_exact_f32(
            _ptr(values),
            n_rows,
            heads,
            head_dim,
            _ptr(w),
            ctypes.c_float(float(eps)),
            _ptr(out),
        )
        if rc != 0:
            raise RuntimeError(f"native head RMSNorm-many rc={rc}")
        self.calls += 1
        self.token_rows += n_rows
        self.head_rows += n_rows * heads
        self.values += n_rows * width
        flat = out.tolist()
        return [flat[r * width : (r + 1) * width] for r in range(n_rows)]

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "token_rows": self.token_rows,
            "head_rows": self.head_rows,
            "values": self.values,
        }
