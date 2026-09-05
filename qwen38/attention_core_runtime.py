#!/usr/bin/env python3
"""ctypes wrapper for the bitwise-exact native causal attention core."""
from __future__ import annotations

from array import array
import ctypes
from pathlib import Path
from typing import Sequence


class ExactAttentionCore:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        fp = ctypes.POINTER(ctypes.c_float)
        fn = self.lib.qwen_attention_core_f32_exact
        fn.argtypes = [
            fp,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            fp,
            fp,
            ctypes.c_size_t,
            ctypes.c_double,
            fp,
        ]
        fn.restype = ctypes.c_int
        self.fn = fn
        self.calls = 0
        self.context_rows = 0
        self.q_values = 0
        self._flat: dict[int, dict[str, object]] = {}

    @staticmethod
    def _ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    def _sync_cache(self, layer: int, cache, kv_dim: int) -> tuple[array, array, int]:
        n_ctx = len(cache["k"])
        if n_ctx != len(cache["v"]) or n_ctx <= 0:
            raise ValueError("attention K/V cache length mismatch")
        newest_k = cache["k"][-1]
        newest_v = cache["v"][-1]
        if len(newest_k) != kv_dim or len(newest_v) != kv_dim:
            raise ValueError("attention K/V cache width mismatch")

        slot = self._flat.get(int(layer))
        if slot is None or int(slot["rows"]) != n_ctx - 1:
            kflat = array("f")
            vflat = array("f")
            for row in cache["k"]:
                if len(row) != kv_dim:
                    raise ValueError("attention K cache width mismatch")
                kflat.extend(map(float, row))
            for row in cache["v"]:
                if len(row) != kv_dim:
                    raise ValueError("attention V cache width mismatch")
                vflat.extend(map(float, row))
            slot = {"k": kflat, "v": vflat, "rows": n_ctx}
            self._flat[int(layer)] = slot
        else:
            kflat = slot["k"]
            vflat = slot["v"]
            assert isinstance(kflat, array) and isinstance(vflat, array)
            kflat.extend(map(float, newest_k))
            vflat.extend(map(float, newest_v))
            slot["rows"] = n_ctx

        kflat = slot["k"]
        vflat = slot["v"]
        assert isinstance(kflat, array) and isinstance(vflat, array)
        if len(kflat) != n_ctx * kv_dim or len(vflat) != n_ctx * kv_dim:
            raise RuntimeError("native attention flat-cache bookkeeping mismatch")
        return kflat, vflat, n_ctx

    def compute(
        self,
        layer: int,
        q: Sequence[float],
        cache,
        *,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        scale: float,
    ) -> list[float]:
        q_dim = int(q_heads) * int(head_dim)
        kv_dim = int(kv_heads) * int(head_dim)
        if len(q) != q_dim:
            raise ValueError(f"native attention Q width={len(q)} expected={q_dim}")

        qbuf = array("f", map(float, q))
        kflat, vflat, n_ctx = self._sync_cache(int(layer), cache, kv_dim)
        out = (ctypes.c_float * q_dim)()
        rc = self.fn(
            self._ptr(qbuf),
            int(q_heads),
            int(kv_heads),
            int(head_dim),
            self._ptr(kflat),
            self._ptr(vflat),
            int(n_ctx),
            float(scale),
            out,
        )
        if rc != 0:
            raise RuntimeError(f"exact native attention core rc={rc}")
        self.calls += 1
        self.context_rows += n_ctx
        self.q_values += q_dim
        return [float(out[i]) for i in range(q_dim)]

    def report(self) -> dict[str, int]:
        aux_bytes = 0
        for slot in self._flat.values():
            k = slot["k"]
            v = slot["v"]
            assert isinstance(k, array) and isinstance(v, array)
            aux_bytes += (len(k) + len(v)) * k.itemsize
        return {
            "calls": self.calls,
            "context_rows": self.context_rows,
            "q_values": self.q_values,
            "probe_aux_cache_bytes_f32": aux_bytes,
        }
