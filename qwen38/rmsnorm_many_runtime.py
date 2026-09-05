#!/usr/bin/env python3
"""Exact sequential many-row RMSNorm bridge for current-best Qwen3.8."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


def load_rmsnorm_many_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.qwen38_rmsnorm_many_exact_f32.argtypes = [
        fp, ctypes.c_size_t, ctypes.c_size_t, fp, ctypes.c_float, fp,
    ]
    lib.qwen38_rmsnorm_many_exact_f32.restype = ctypes.c_int
    return lib


def _ptr(buf: array):
    return (ctypes.c_float * len(buf)).from_buffer(buf)


class ExactRMSNormMany:
    def __init__(self, lib_path: Path):
        self.lib = load_rmsnorm_many_lib(lib_path)
        self.calls = 0
        self.rows = 0
        self.values = 0

    def compute(
        self,
        rows: Sequence[Sequence[float]],
        weight: Sequence[float],
        eps: float,
    ) -> list[list[float]]:
        n_rows = len(rows)
        if n_rows == 0:
            return []
        width = len(weight)
        if width == 0:
            raise ValueError("RMSNorm-many empty weight")
        values = array("f")
        for row in rows:
            if len(row) != width:
                raise ValueError(
                    f"RMSNorm-many row width={len(row)} expected={width}")
            values.extend(row)
        w = array("f", weight)
        out = array("f", [0.0]) * (n_rows * width)
        rc = self.lib.qwen38_rmsnorm_many_exact_f32(
            _ptr(values), n_rows, width, _ptr(w),
            ctypes.c_float(float(eps)), _ptr(out))
        if rc != 0:
            raise RuntimeError(f"native RMSNorm-many rc={rc}")
        self.calls += 1
        self.rows += n_rows
        self.values += n_rows * width
        flat = out.tolist()
        return [flat[i * width : (i + 1) * width] for i in range(n_rows)]

    def report(self) -> dict[str, int]:
        return {"calls": self.calls, "rows": self.rows, "values": self.values}
