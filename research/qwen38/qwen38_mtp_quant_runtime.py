#!/usr/bin/env python3
"""MTP-only quant runtime extension for Qwen3.8.

The proven decoder QuantRuntime remains unchanged.  This adapter keeps the
proven Q6_K/Q8_0 AVX2 library as the base runtime and binds the isolated Q4_0
library only for blk.64 matrices.  That avoids silently sending the MTP's
shared Q8_0 LM head through the older scalar bridge.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn


def load_q4_native(path: Path):
    lib = ctypes.CDLL(str(path))
    c_u8p = ctypes.POINTER(ctypes.c_uint8)
    c_fp = ctypes.POINTER(ctypes.c_float)
    for name in ("qwen_matvec_q4_0_q8_0_reference", "qwen_matvec_q4_0_q8_0_scalar"):
        fn = getattr(lib, name)
        fn.argtypes = [c_u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                       c_u8p, ctypes.c_size_t, c_fp]
        fn.restype = ctypes.c_int
    return lib


class MTPQuantRuntime(gdn.QuantRuntime):
    def __init__(self, base_lib, q4_lib, *, q4_reference: bool = False):
        super().__init__(base_lib)
        self.q4_lib = q4_lib
        self.q4_reference = bool(q4_reference)
        self.q4_matvec_rows = 0
        self.q4_weight_bytes = 0

    def quantize(self, x: Sequence[float], kind: str):
        # GGML Q4_0 dot products consume Q8_0 activations.
        if kind == "Q4_0":
            return super().quantize(x, "Q8_0")
        return super().quantize(x, kind)

    def matvec(self, weights: memoryview, meta: dict[str, Any], x: Sequence[float], prepared=None) -> list[float]:
        kind = meta["type_name"]
        if kind != "Q4_0":
            return super().matvec(weights, meta, x, prepared=prepared)

        ne0, rows = map(int, meta["shape"])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: x={len(x)} ne0={ne0}")
        if prepared is None:
            prepared = self.quantize(x, "Q4_0")
        activation, activation_bytes = prepared
        w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
        out = (ctypes.c_float * rows)()
        fn = (self.q4_lib.qwen_matvec_q4_0_q8_0_reference if self.q4_reference
              else self.q4_lib.qwen_matvec_q4_0_q8_0_scalar)
        rc = fn(w_arr, len(weights), rows, ne0, activation, activation_bytes, out)
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: Q4 native matvec failed rc={rc}")
        self.matvec_rows += rows
        self.q4_matvec_rows += rows
        self.q4_weight_bytes += len(weights)
        return [float(out[i]) for i in range(rows)]

    def report(self) -> dict[str, Any]:
        return {
            "q4_reference": self.q4_reference,
            "activation_quantizations": self.activation_quantizations,
            "matvec_rows": self.matvec_rows,
            "q4_matvec_rows": self.q4_matvec_rows,
            "q4_weight_bytes": self.q4_weight_bytes,
        }


def sanity() -> None:
    assert issubclass(MTPQuantRuntime, gdn.QuantRuntime)
    print({"schema": "qwen38-mtp-quant-runtime-sanity-v1", "status": "PASS"})


if __name__ == "__main__":
    sanity()
