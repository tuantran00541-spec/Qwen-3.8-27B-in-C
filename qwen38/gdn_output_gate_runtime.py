#!/usr/bin/env python3
"""ctypes wrapper for exact native GDN output RMSNorm + SiLU gating."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactGDNOutputGate:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        fp = ctypes.POINTER(ctypes.c_float)
        fn = self.lib.qwen_gdn_output_rmsnorm_gate_f32_exact
        fn.argtypes = [
            fp,
            fp,
            fp,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            fp,
        ]
        fn.restype = ctypes.c_int
        self.fn = fn
        self.calls = 0
        self.values = 0

    @staticmethod
    def _ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    def compute(
        self,
        core,
        z: Sequence[float],
        weight: Sequence[float],
        *,
        heads: int,
        head_dim: int,
        eps: float,
    ) -> list[float]:
        n = int(heads) * int(head_dim)
        if len(core) != n or len(z) != n or len(weight) != int(head_dim):
            raise ValueError(
                f"GDN output gate shape mismatch core={len(core)} z={len(z)} "
                f"weight={len(weight)} expected={n}/{head_dim}"
            )

        core_keepalive = None
        if isinstance(core, ctypes.Array):
            core_ptr = ctypes.cast(core, ctypes.POINTER(ctypes.c_float))
        else:
            core_keepalive = array("f", map(float, core))
            core_ptr = self._ptr(core_keepalive)
        zbuf = array("f", map(float, z))
        wbuf = array("f", map(float, weight))
        out = (ctypes.c_float * n)()
        rc = self.fn(
            core_ptr,
            self._ptr(zbuf),
            self._ptr(wbuf),
            int(heads),
            int(head_dim),
            float(eps),
            out,
        )
        if rc != 0:
            raise RuntimeError(f"exact native GDN output gate rc={rc}")
        self.calls += 1
        self.values += n
        return [float(out[i]) for i in range(n)]

    def report(self) -> dict[str, int]:
        return {"calls": self.calls, "values": self.values}
