#!/usr/bin/env python3
"""ctypes wrapper for exact Qwen3.8 GDN causal-conv + SiLU probe kernel."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactGDNConvSilu:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        fp = ctypes.POINTER(ctypes.c_float)
        fn = self.lib.qwen_gdn_conv_silu_many_f32_exact
        fn.argtypes = [
            fp,
            ctypes.c_size_t,
            ctypes.c_size_t,
            fp,
            fp,
            ctypes.c_size_t,
            fp,
        ]
        fn.restype = ctypes.c_int
        self.fn = fn
        self.calls = 0
        self.tokens = 0
        self.values = 0

    @staticmethod
    def _ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    def compute(
        self,
        qkv_rows: Sequence[Sequence[float]],
        kernels: Sequence[float],
        history: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        n_tokens = len(qkv_rows)
        if n_tokens <= 0:
            raise ValueError("native GDN conv requires at least one token")
        conv_dim = len(qkv_rows[0])
        if conv_dim <= 0 or any(len(row) != conv_dim for row in qkv_rows):
            raise ValueError("GDN qkv row width mismatch")
        if len(kernels) != conv_dim * 4:
            raise ValueError("GDN conv kernel width mismatch")
        if len(history) > 3 or any(len(row) != conv_dim for row in history):
            raise ValueError("GDN conv history shape mismatch")

        qkv = array("f", (float(v) for row in qkv_rows for v in row))
        weights = array("f", map(float, kernels))
        hist = array("f", (float(v) for row in history for v in row))
        out = array("f", [0.0]) * (n_tokens * conv_dim)
        null_fp = ctypes.POINTER(ctypes.c_float)()
        hist_ptr = self._ptr(hist) if hist else null_fp
        rc = self.fn(
            self._ptr(qkv),
            n_tokens,
            conv_dim,
            self._ptr(weights),
            hist_ptr,
            len(history),
            self._ptr(out),
        )
        if rc != 0:
            raise RuntimeError(f"exact native GDN causal conv rc={rc}")

        self.calls += 1
        self.tokens += n_tokens
        self.values += n_tokens * conv_dim
        return [list(out[j * conv_dim:(j + 1) * conv_dim]) for j in range(n_tokens)]

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "values": self.values,
        }
