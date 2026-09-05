#!/usr/bin/env python3
"""ctypes wrapper for exact Qwen3.8 staged-prefill residual addition."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactResidualAdd:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        fp = ctypes.POINTER(ctypes.c_float)
        self.lib.qwen38_residual_add_many_exact_f32.argtypes = [
            fp, fp, ctypes.c_size_t, ctypes.c_size_t, fp,
        ]
        self.lib.qwen38_residual_add_many_exact_f32.restype = ctypes.c_int
        self.calls = 0
        self.rows = 0
        self.values = 0

    @staticmethod
    def _ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    @staticmethod
    def _flatten(rows: Sequence[Sequence[float]], width: int) -> array:
        out = array("f")
        for row in rows:
            if len(row) != width:
                raise ValueError("residual-add ragged rows")
            out.extend(map(float, row))
        return out

    def compute(
        self,
        a_rows: Sequence[Sequence[float]],
        b_rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        rows = len(a_rows)
        if rows == 0 or len(b_rows) != rows:
            raise ValueError("residual-add row mismatch")
        width = len(a_rows[0])
        if width == 0:
            raise ValueError("residual-add empty width")

        a = self._flatten(a_rows, width)
        b = self._flatten(b_rows, width)
        out = array("f", [0.0]) * (rows * width)
        rc = self.lib.qwen38_residual_add_many_exact_f32(
            self._ptr(a), self._ptr(b), rows, width, self._ptr(out))
        if rc != 0:
            raise RuntimeError(f"native residual-add rc={rc}")

        self.calls += 1
        self.rows += rows
        self.values += rows * width
        return [out[j * width:(j + 1) * width].tolist() for j in range(rows)]

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "rows": self.rows,
            "values": self.values,
        }
