#!/usr/bin/env python3
"""ctypes wrapper for exact Qwen3.8 full-attention sigmoid gating."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactAttentionGate:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        fp = ctypes.POINTER(ctypes.c_float)
        self.lib.qwen38_attention_gate_exact_f32.argtypes = [
            fp, fp, ctypes.c_size_t, fp,
        ]
        self.lib.qwen38_attention_gate_exact_f32.restype = ctypes.c_int
        self.calls = 0
        self.values = 0

    @staticmethod
    def _ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    def compute(
        self,
        pregate: Sequence[float],
        gate: Sequence[float],
    ) -> list[float]:
        n = len(pregate)
        if n == 0 or len(gate) != n:
            raise ValueError("attention-gate length mismatch")

        p = array("f", map(float, pregate))
        g = array("f", map(float, gate))
        out = array("f", [0.0]) * n
        rc = self.lib.qwen38_attention_gate_exact_f32(
            self._ptr(p), self._ptr(g), n, self._ptr(out))
        if rc != 0:
            raise RuntimeError(f"native attention-gate rc={rc}")

        self.calls += 1
        self.values += n
        return out.tolist()

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "values": self.values,
        }
