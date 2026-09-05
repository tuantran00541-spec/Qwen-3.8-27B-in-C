#!/usr/bin/env python3
"""ctypes wrapper for exact Qwen3.8 GDN q/k repeat and q scaling."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactGDNRepeatScale:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        fp = ctypes.POINTER(ctypes.c_float)
        self.lib.qwen38_gdn_repeat_scale_many_exact_f32.argtypes = [
            fp, fp, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, fp, fp,
        ]
        self.lib.qwen38_gdn_repeat_scale_many_exact_f32.restype = ctypes.c_int
        self.calls = 0
        self.rows = 0
        self.q_values = 0
        self.k_values = 0

    @staticmethod
    def _ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    @staticmethod
    def _flatten(rows: Sequence[Sequence[float]], width: int) -> array:
        out = array("f")
        for row in rows:
            if len(row) != width:
                raise ValueError("GDN repeat-scale ragged rows")
            out.extend(map(float, row))
        return out

    def compute(
        self,
        q_rows: Sequence[Sequence[float]],
        k_rows: Sequence[Sequence[float]],
        *,
        repeats: int,
        scale: float,
    ) -> tuple[list[list[float]], list[list[float]]]:
        rows = len(q_rows)
        if rows == 0 or len(k_rows) != rows:
            raise ValueError("GDN repeat-scale row mismatch")
        key_dim = len(q_rows[0])
        if key_dim == 0 or repeats <= 0:
            raise ValueError("GDN repeat-scale invalid shape")

        q = self._flatten(q_rows, key_dim)
        k = self._flatten(k_rows, key_dim)
        out_width = key_dim * repeats
        q_out = array("f", [0.0]) * (rows * out_width)
        k_out = array("f", [0.0]) * (rows * out_width)
        rc = self.lib.qwen38_gdn_repeat_scale_many_exact_f32(
            self._ptr(q), self._ptr(k), rows, key_dim, repeats,
            ctypes.c_float(float(scale)), self._ptr(q_out), self._ptr(k_out))
        if rc != 0:
            raise RuntimeError(f"native GDN repeat-scale rc={rc}")

        self.calls += 1
        self.rows += rows
        self.q_values += rows * out_width
        self.k_values += rows * out_width
        q_result = [q_out[j * out_width:(j + 1) * out_width].tolist() for j in range(rows)]
        k_result = [k_out[j * out_width:(j + 1) * out_width].tolist() for j in range(rows)]
        return q_result, k_result

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "rows": self.rows,
            "q_values": self.q_values,
            "k_values": self.k_values,
        }
