#!/usr/bin/env python3
"""ctypes wrapper for the exact scalar Qwen3.8 RMSNorm kernel."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactRMSNorm:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        self.lib.qwen38_rmsnorm_exact_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.qwen38_rmsnorm_exact_f32.restype = ctypes.c_int
        self.lib.qwen38_rmsnorm_heads_exact_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.qwen38_rmsnorm_heads_exact_f32.restype = ctypes.c_int
        self.calls = 0
        self.rows = 0
        self.values = 0
        self.head_calls = 0
        self.head_rows = 0
        self.head_values = 0

    @staticmethod
    def _buffer(values: Sequence[float]) -> array:
        return array("f", map(float, values))

    @staticmethod
    def _ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    def compute(self, values: Sequence[float], weight: Sequence[float], eps: float) -> list[float]:
        width = len(values)
        if width == 0 or len(weight) != width:
            raise ValueError("RMSNorm shape mismatch")
        x = self._buffer(values)
        w = self._buffer(weight)
        out = array("f", [0.0]) * width
        rc = self.lib.qwen38_rmsnorm_exact_f32(
            self._ptr(x), self._ptr(w), width, ctypes.c_float(float(eps)), self._ptr(out))
        if rc != 0:
            raise RuntimeError(f"native RMSNorm rc={rc}")
        self.calls += 1
        self.rows += 1
        self.values += width
        return out.tolist()

    def compute_heads(
        self,
        values: Sequence[float],
        heads: int,
        weight: Sequence[float],
        eps: float,
    ) -> list[float]:
        heads = int(heads)
        if heads <= 0 or not weight:
            raise ValueError("invalid RMSNorm-heads shape")
        head_dim = len(weight)
        if len(values) != heads * head_dim:
            raise ValueError("RMSNorm-heads shape mismatch")
        x = self._buffer(values)
        w = self._buffer(weight)
        out = array("f", [0.0]) * len(values)
        rc = self.lib.qwen38_rmsnorm_heads_exact_f32(
            self._ptr(x), heads, head_dim, self._ptr(w), ctypes.c_float(float(eps)), self._ptr(out))
        if rc != 0:
            raise RuntimeError(f"native RMSNorm-heads rc={rc}")
        self.head_calls += 1
        self.head_rows += heads
        self.head_values += len(values)
        return out.tolist()

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "rows": self.rows,
            "values": self.values,
            "head_calls": self.head_calls,
            "head_rows": self.head_rows,
            "head_values": self.head_values,
            "total_rows": self.rows + self.head_rows,
            "total_values": self.values + self.head_values,
        }
