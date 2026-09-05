#!/usr/bin/env python3
"""Reusable exact native-F32 runtime wrapper for Qwen3.8 research paths.

The base QuantRuntime remains authoritative for Q6_K/Q8_0 tensors.  This module
only replaces F32 matrix matvec with ``qwen_matvec_f32_fsum_exact`` from
``f32_fsum_matvec.c``.  That kernel has synthetic double-bitwise parity with
CPython ``math.fsum`` and a real 11-token layer-major prefill gate with exact
hidden/persistent-state parity.

Keep this opt-in until the integrated generator/prefill gates are green.  The
wrapper deliberately exposes the same ``quantize``/``matvec`` surface used by
the existing exact runtime so callers do not need a second decoder path.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from typing import Any, Sequence


def load_f32_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    fp = ctypes.POINTER(ctypes.c_float)
    dp = ctypes.POINTER(ctypes.c_double)
    lib.qwen_matvec_f32_fsum_exact.argtypes = [
        fp, ctypes.c_size_t, ctypes.c_size_t, dp, dp,
    ]
    lib.qwen_matvec_f32_fsum_exact.restype = ctypes.c_int
    return lib


class NativeF32Runtime:
    """Delegate quantized work unchanged and accelerate only F32 matrices."""

    def __init__(self, runtime, f32_lib):
        self.runtime = runtime
        self.f32_lib = f32_lib
        self.native_f32_calls = 0
        self.native_f32_rows = 0
        self.native_f32_terms = 0

    def quantize(self, x: Sequence[float], kind: str):
        return self.runtime.quantize(x, kind)

    @property
    def activation_quantizations(self):
        return self.runtime.activation_quantizations

    @property
    def matvec_rows(self):
        return self.runtime.matvec_rows

    def matvec(
        self,
        weights: memoryview,
        meta: dict[str, Any],
        x: Sequence[float],
        prepared=None,
    ):
        if meta["type_name"] != "F32":
            return self.runtime.matvec(weights, meta, x, prepared=prepared)

        if sys.byteorder != "little":
            raise RuntimeError("native F32 runtime requires little-endian host")
        ne0, rows = map(int, meta["shape"])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: x={len(x)} ne0={ne0}")
        expected_bytes = ne0 * rows * 4
        if len(weights) != expected_bytes:
            raise ValueError(
                f"{meta['name']}: F32 bytes={len(weights)} expected={expected_bytes}"
            )

        # K3 ring slots are mutable aligned buffers.  from_buffer keeps the F32
        # path zero-copy and reads the exact currently-bound tensor bytes.
        w_arr = (ctypes.c_float * (ne0 * rows)).from_buffer(weights)
        x_arr = (ctypes.c_double * ne0)(*map(float, x))
        out = (ctypes.c_double * rows)()
        rc = self.f32_lib.qwen_matvec_f32_fsum_exact(
            w_arr, rows, ne0, x_arr, out
        )
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: native F32 fsum matvec rc={rc}")

        self.native_f32_calls += 1
        self.native_f32_rows += rows
        self.native_f32_terms += rows * ne0
        return [float(out[i]) for i in range(rows)]

    def report(self) -> dict[str, int]:
        return {
            "native_f32_calls": self.native_f32_calls,
            "native_f32_rows": self.native_f32_rows,
            "native_f32_terms": self.native_f32_terms,
        }


def enable_native_f32(engine, f32_lib_path: Path) -> NativeF32Runtime:
    """Install the exact F32 wrapper on an existing StatefulK3Generator."""
    if isinstance(engine.runtime, NativeF32Runtime):
        return engine.runtime
    wrapped = NativeF32Runtime(engine.runtime, load_f32_lib(f32_lib_path))
    engine.runtime = wrapped
    return wrapped
