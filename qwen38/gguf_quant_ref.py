#!/usr/bin/env python3
"""Small, dependency-free GGML quant reference helpers for the Qwen3.8 lab.

This module is intentionally a correctness oracle, not a fast inference kernel.
The Q8_0 and Q6_K decode rules mirror llama.cpp/ggml's reference
``dequantize_row_q8_0`` and ``dequantize_row_q6_K`` implementations at the
runtime-baseline pin 557614e0296ff4a5b6f649737a65ae2076eea2fd.

GGML/GGUF matrix shapes are interpreted as ``[ne0, ne1]`` where ``ne0`` is the
row width (the inner/matvec dimension) and ``ne1`` is the number of rows.
"""
from __future__ import annotations

import math
import struct
from typing import Sequence

QK8_0 = 32
BLOCK_Q8_0 = 34  # fp16 scale + 32 int8 quants
QK_Q6_K = 256
BLOCK_Q6_K = 210  # 128 ql + 64 qh + 16 int8 scales + fp16 super-scale


def _need_exact(data: bytes | bytearray | memoryview, expected: int, label: str) -> memoryview:
    view = memoryview(data).cast("B")
    if len(view) != expected:
        raise ValueError(f"{label}: expected {expected} bytes, got {len(view)}")
    return view


def row_nbytes(type_name: str, ne0: int) -> int:
    """Return the packed GGML byte size of one row for supported reference types."""
    ne0 = int(ne0)
    if ne0 <= 0:
        raise ValueError("ne0 must be positive")
    if type_name == "Q8_0":
        if ne0 % QK8_0:
            raise ValueError("Q8_0 ne0 must be divisible by 32")
        return (ne0 // QK8_0) * BLOCK_Q8_0
    if type_name == "Q6_K":
        if ne0 % QK_Q6_K:
            raise ValueError("Q6_K ne0 must be divisible by 256")
        return (ne0 // QK_Q6_K) * BLOCK_Q6_K
    if type_name == "F32":
        return ne0 * 4
    raise ValueError(f"unsupported reference type: {type_name}")


def dequantize_q8_0(data: bytes | bytearray | memoryview, ne0: int) -> list[float]:
    """Decode one Q8_0 GGML row into Python floats."""
    packed = _need_exact(data, row_nbytes("Q8_0", ne0), "Q8_0 row")
    out = [0.0] * int(ne0)
    dst = 0
    for base in range(0, len(packed), BLOCK_Q8_0):
        d = float(struct.unpack_from("<e", packed, base)[0])
        quants = struct.unpack_from("<32b", packed, base + 2)
        for q in quants:
            out[dst] = d * q
            dst += 1
    return out


def dequantize_q6_k(data: bytes | bytearray | memoryview, ne0: int) -> list[float]:
    """Decode one Q6_K GGML row exactly following ggml's scalar reference layout."""
    packed = _need_exact(data, row_nbytes("Q6_K", ne0), "Q6_K row")
    out = [0.0] * int(ne0)
    block_out = 0

    for base in range(0, len(packed), BLOCK_Q6_K):
        ql = packed[base : base + 128]
        qh = packed[base + 128 : base + 192]
        scales = struct.unpack_from("<16b", packed, base + 192)
        d = float(struct.unpack_from("<e", packed, base + 208)[0])

        ql_base = 0
        qh_base = 0
        sc_base = 0
        for n in (0, 128):
            for l in range(32):
                iscale = l // 16
                low0 = ql[ql_base + l]
                low1 = ql[ql_base + l + 32]
                high = qh[qh_base + l]

                q1 = ((low0 & 0x0F) | (((high >> 0) & 0x03) << 4)) - 32
                q2 = ((low1 & 0x0F) | (((high >> 2) & 0x03) << 4)) - 32
                q3 = ((low0 >> 4) | (((high >> 4) & 0x03) << 4)) - 32
                q4 = ((low1 >> 4) | (((high >> 6) & 0x03) << 4)) - 32

                out[block_out + n + l + 0] = d * scales[sc_base + iscale + 0] * q1
                out[block_out + n + l + 32] = d * scales[sc_base + iscale + 2] * q2
                out[block_out + n + l + 64] = d * scales[sc_base + iscale + 4] * q3
                out[block_out + n + l + 96] = d * scales[sc_base + iscale + 6] * q4

            ql_base += 64
            qh_base += 32
            sc_base += 8
        block_out += QK_Q6_K

    return out


def decode_row(type_name: str, data: bytes | bytearray | memoryview, ne0: int) -> list[float]:
    if type_name == "Q8_0":
        return dequantize_q8_0(data, ne0)
    if type_name == "Q6_K":
        return dequantize_q6_k(data, ne0)
    if type_name == "F32":
        packed = _need_exact(data, row_nbytes("F32", ne0), "F32 row")
        return [float(x) for x in struct.unpack(f"<{int(ne0)}f", packed)]
    raise ValueError(f"unsupported reference type: {type_name}")


def matvec_rows(
    data: bytes | bytearray | memoryview,
    *,
    type_name: str,
    shape: Sequence[int],
    vector: Sequence[float],
    row_start: int = 0,
    row_count: int | None = None,
) -> list[float]:
    """Reference GGML matrix-vector product for a contiguous 2-D tensor.

    ``shape`` must be ``[ne0, ne1]``. This is deliberately row-at-a-time so it
    never materializes the whole dequantized matrix; it is suitable for small
    semantic slices and test oracles, not production inference.
    """
    if len(shape) != 2:
        raise ValueError("reference matvec expects a 2-D GGML shape [ne0, ne1]")
    ne0, ne1 = (int(shape[0]), int(shape[1]))
    if len(vector) != ne0:
        raise ValueError(f"vector length {len(vector)} does not match ne0={ne0}")
    row_start = int(row_start)
    if row_start < 0 or row_start > ne1:
        raise ValueError("row_start out of range")
    if row_count is None:
        row_count = ne1 - row_start
    row_count = int(row_count)
    if row_count < 0 or row_start + row_count > ne1:
        raise ValueError("row_count out of range")

    stride = row_nbytes(type_name, ne0)
    packed = _need_exact(data, stride * ne1, f"{type_name} matrix")
    x = tuple(float(v) for v in vector)
    result: list[float] = []
    for row in range(row_start, row_start + row_count):
        start = row * stride
        values = decode_row(type_name, packed[start : start + stride], ne0)
        result.append(math.fsum(a * b for a, b in zip(values, x)))
    return result
