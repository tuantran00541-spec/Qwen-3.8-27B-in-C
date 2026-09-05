#!/usr/bin/env python3
"""Exact activation-quantization input marshaling candidate for Qwen3.8.

The native Q8_K / Q8_0 quantizers are unchanged.  This helper only replaces
``(ctypes.c_float * n)(*map(float, x))`` with one contiguous ``array('f')`` and
a zero-copy ctypes view over that F32 buffer before calling the same C function.
"""
from __future__ import annotations

from array import array
import ctypes
import sys
import time
from typing import Sequence


class FastActivationQuantizer:
    def __init__(self, base_runtime):
        if sys.byteorder != "little":
            raise RuntimeError("fast activation quantizer requires little-endian host")
        if not hasattr(base_runtime, "lib"):
            raise TypeError("expected base QuantRuntime with native quantizer library")
        self.base = base_runtime
        self.calls = 0
        self.values = 0
        self.seconds = 0.0
        self._original_quantize = base_runtime.quantize

    def quantize(self, x: Sequence[float], kind: str):
        t0 = time.monotonic()
        n = len(x)
        values = array("f", x)
        x_arr = (ctypes.c_float * n).from_buffer(values)
        if kind == "Q6_K":
            if n % 256:
                raise ValueError("Q6_K activation width must be divisible by 256")
            nbytes = (n // 256) * 292
            fn = self.base.lib.qwen_quantize_q8_k_scalar
        elif kind == "Q8_0":
            if n % 32:
                raise ValueError("Q8_0 activation width must be divisible by 32")
            nbytes = (n // 32) * 34
            fn = self.base.lib.qwen_quantize_q8_0_scalar
        else:
            raise ValueError(kind)
        buf = (ctypes.c_uint8 * nbytes)()
        rc = fn(x_arr, n, buf, nbytes)
        if rc != 0:
            raise RuntimeError(f"activation quantization {kind} failed rc={rc}")
        self.base.activation_quantizations += 1
        self.calls += 1
        self.values += n
        self.seconds += time.monotonic() - t0
        return buf, nbytes

    def install(self) -> None:
        self.base.quantize = self.quantize

    def restore(self) -> None:
        self.base.quantize = self._original_quantize

    def report(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "values": self.values,
            "seconds": self.seconds,
        }


def find_base_quant_runtime(runtime):
    cur = runtime
    seen = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "lib") and hasattr(cur, "activation_quantizations"):
            return cur
        if not hasattr(cur, "runtime"):
            break
        cur = cur.runtime
    raise TypeError("could not find base QuantRuntime")
