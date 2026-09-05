#!/usr/bin/env python3
"""Exact sequential batched GDN-state bridge for current-best Qwen3.8.

The C ABI calls the already-proven qwen_gdn_ar_step_f32 once per row, in order.
This wrapper only amortizes Python/ctypes marshaling and call overhead.
"""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence

import qwen35_gdn_quant_layer_gate as gdn


VALUE_DIM = gdn.VALUE_DIM
HEADS = gdn.V_HEADS


def load_batch_state_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.qwen_gdn_ar_batch_f32.argtypes = [
        fp, ctypes.c_size_t, fp, fp, fp, fp, fp, fp,
    ]
    lib.qwen_gdn_ar_batch_f32.restype = ctypes.c_int
    return lib


def _flat_f32(rows: Sequence[Sequence[float]], width: int, label: str) -> array:
    out = array("f")
    for row in rows:
        if len(row) != width:
            raise ValueError(f"{label} row width={len(row)} expected={width}")
        out.extend(row)
    return out


def _view(buf: array):
    return (ctypes.c_float * len(buf)).from_buffer(buf)


class ExactGDNStateBatch:
    def __init__(self, lib_path: Path):
        self.lib = load_batch_state_lib(lib_path)
        self.calls = 0
        self.rows = 0
        self.q_values = 0
        self.k_values = 0
        self.v_values = 0
        self.gate_values = 0
        self.beta_values = 0
        self.output_values = 0

    def compute(
        self,
        state,
        q_rows: Sequence[Sequence[float]],
        k_rows: Sequence[Sequence[float]],
        v_rows: Sequence[Sequence[float]],
        gate_rows: Sequence[Sequence[float]],
        beta_rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        rows = len(q_rows)
        if not rows:
            return []
        if not (
            len(k_rows) == rows
            and len(v_rows) == rows
            and len(gate_rows) == rows
            and len(beta_rows) == rows
        ):
            raise ValueError("batched GDN state inputs have mismatched row counts")

        q = _flat_f32(q_rows, VALUE_DIM, "q")
        k = _flat_f32(k_rows, VALUE_DIM, "k")
        v = _flat_f32(v_rows, VALUE_DIM, "v")
        gate = _flat_f32(gate_rows, HEADS, "gate")
        beta = _flat_f32(beta_rows, HEADS, "beta")
        out = array("f", [0.0]) * (rows * VALUE_DIM)

        rc = self.lib.qwen_gdn_ar_batch_f32(
            state,
            rows,
            _view(q),
            _view(k),
            _view(v),
            _view(gate),
            _view(beta),
            _view(out),
        )
        if rc != 0:
            raise RuntimeError(f"batched GDN state kernel rc={rc}")

        self.calls += 1
        self.rows += rows
        values = rows * VALUE_DIM
        head_values = rows * HEADS
        self.q_values += values
        self.k_values += values
        self.v_values += values
        self.gate_values += head_values
        self.beta_values += head_values
        self.output_values += values

        flat = out.tolist()
        return [
            flat[i * VALUE_DIM : (i + 1) * VALUE_DIM]
            for i in range(rows)
        ]

    def report(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "rows": self.rows,
            "q_values": self.q_values,
            "k_values": self.k_values,
            "v_values": self.v_values,
            "gate_values": self.gate_values,
            "beta_values": self.beta_values,
            "output_values": self.output_values,
        }
