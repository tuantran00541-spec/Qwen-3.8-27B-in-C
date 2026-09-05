#!/usr/bin/env python3
"""ctypes wrapper for exact Qwen3.8 staged-prefill SwiGLU kernel."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactSwiGLU:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        fp = ctypes.POINTER(ctypes.c_float)
        fn = self.lib.qwen_swiglu_many_f32_exact
        fn.argtypes = [
            fp,
            fp,
            ctypes.c_size_t,
            ctypes.c_size_t,
            fp,
        ]
        fn.restype = ctypes.c_int
        self.fn = fn
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
                raise ValueError("SwiGLU row width mismatch")
            if isinstance(row, list):
                out.fromlist(row)
            else:
                out.extend(row)
        return out

    def compute(
        self,
        gate_rows: Sequence[Sequence[float]],
        up_rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        n_rows = len(gate_rows)
        if n_rows <= 0 or len(up_rows) != n_rows:
            raise ValueError("native SwiGLU requires matching non-empty row sets")
        width = len(gate_rows[0])
        if width <= 0:
            raise ValueError("native SwiGLU requires non-empty rows")

        gate = self._flatten(gate_rows, width)
        up = self._flatten(up_rows, width)
        out = array("f", [0.0]) * (n_rows * width)
        rc = self.fn(
            self._ptr(gate),
            self._ptr(up),
            n_rows,
            width,
            self._ptr(out),
        )
        if rc != 0:
            raise RuntimeError(f"exact native SwiGLU rc={rc}")

        self.calls += 1
        self.rows += n_rows
        self.values += n_rows * width
        return [
            list(out[j * width:(j + 1) * width])
            for j in range(n_rows)
        ]

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "rows": self.rows,
            "values": self.values,
        }
