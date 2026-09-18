#!/usr/bin/env python3
"""Native C runtime adapter for Prism Bonsai 2 low-bit GGUF weights.

This module intentionally exposes only the arithmetic required to validate the
new K3 streaming path first: Hadamard/sign activation preparation, Q8_0
activation quantization, and PTQ1_0/PQ2_0 matvec. Full Qwen3.8 generation is
wired only after these primitives pass real-weight gates.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Mapping, Sequence


_C_U8P = ctypes.POINTER(ctypes.c_uint8)
_C_I8P = ctypes.POINTER(ctypes.c_int8)
_C_FP = ctypes.POINTER(ctypes.c_float)


def _load_sign_table(metadata: Mapping[str, object]) -> dict[int, tuple[int, ...]]:
    mode = str(metadata.get("prism.hadamard.sign_mode", "identity"))
    if mode == "identity":
        return {}
    if mode != "explicit":
        raise ValueError(f"unsupported prism.hadamard.sign_mode={mode!r}")

    widths = [int(v) for v in metadata.get("prism.hadamard.sign_widths", [])]
    values = [int(v) for v in metadata.get("prism.hadamard.sign_values", [])]
    if not widths:
        raise ValueError("explicit Hadamard sign mode has no sign widths")

    out: dict[int, tuple[int, ...]] = {}
    off = 0
    for width in widths:
        if width <= 0 or off + width > len(values):
            raise ValueError(f"invalid Hadamard sign width {width}")
        row = tuple(values[off : off + width])
        if any(v not in (-1, 1) for v in row):
            raise ValueError(f"Hadamard sign width {width} contains values other than +/-1")
        if width in out:
            raise ValueError(f"duplicate Hadamard sign width {width}")
        out[width] = row
        off += width
    if off != len(values):
        raise ValueError("Hadamard sign_values length does not match sign_widths")
    return out


class Bonsai2NativeRuntime:
    def __init__(self, library: Path, metadata: Mapping[str, object]) -> None:
        self.library = Path(library)
        self.lib = ctypes.CDLL(str(self.library))

        self.lib.qwen_quantize_q8_0_scalar.argtypes = [
            _C_FP, ctypes.c_size_t, _C_U8P, ctypes.c_size_t
        ]
        self.lib.qwen_quantize_q8_0_scalar.restype = ctypes.c_int

        for name in (
            "qwen_bonsai2_matvec_ptq1_0_q8_0",
            "qwen_bonsai2_matvec_pq2_0_q8_0",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [
                _C_U8P,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                _C_U8P,
                ctypes.c_size_t,
                _C_FP,
            ]
            fn.restype = ctypes.c_int

        self.lib.qwen_bonsai2_fwht_blocks.argtypes = [
            _C_FP, ctypes.c_size_t, ctypes.c_size_t, _C_I8P
        ]
        self.lib.qwen_bonsai2_fwht_blocks.restype = ctypes.c_int

        version = int(metadata.get("prism.hadamard.version", 0))
        if version != 1:
            raise ValueError(f"Bonsai 2 requires prism.hadamard.version=1, got {version}")
        self.block_size = int(metadata.get("prism.hadamard.block_size", 0))
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError(f"invalid Hadamard block size {self.block_size}")
        if metadata.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard":
            raise ValueError("unsupported Bonsai 2 Hadamard transform")
        if metadata.get("prism.hadamard.axis") != "input-last-dimension":
            raise ValueError("unsupported Bonsai 2 Hadamard axis")

        self.folded_weights = frozenset(
            str(v) for v in metadata.get("prism.hadamard.weight_names", [])
        )
        self.inverse_weights = frozenset(
            str(v) for v in metadata.get("prism.hadamard.inverse_weight_names", [])
        )
        self.sign_mode = str(metadata.get("prism.hadamard.sign_mode", "identity"))
        self.signs_by_width = _load_sign_table(metadata)
        self.gdn_v_grouped = bool(metadata.get("prism.hadamard.gdn_v_grouped", False))

        self.activation_quantizations = 0
        self.hadamard_transforms = 0
        self.matvec_rows = 0

    def _sign_array(self, width: int):
        if self.sign_mode == "identity":
            return None, None
        try:
            values = self.signs_by_width[int(width)]
        except KeyError as exc:
            raise ValueError(f"no explicit Hadamard sign vector for width {width}") from exc
        arr = (ctypes.c_int8 * width)(*values)
        return arr, ctypes.cast(arr, _C_I8P)

    def transform_activation(self, weight_name: str, x: Sequence[float]):
        n = len(x)
        arr = (ctypes.c_float * n)(*map(float, x))
        if weight_name not in self.folded_weights:
            return arr

        if self.gdn_v_grouped and ".ssm_out." in weight_name:
            raise NotImplementedError(
                "GDN ssm_out requires the Prism tiled->grouped head permutation "
                "before Hadamard; the first native K3 gate deliberately excludes it"
            )

        sign_owner, sign_ptr = self._sign_array(n)
        del sign_owner  # pointer is consumed synchronously by the native call
        rc = self.lib.qwen_bonsai2_fwht_blocks(arr, n, self.block_size, sign_ptr)
        if rc != 0:
            raise RuntimeError(f"{weight_name}: native Hadamard failed rc={rc}")
        self.hadamard_transforms += 1
        return arr

    def quantize_q8_0(self, arr, n: int):
        if n % 32:
            raise ValueError(f"Q8_0 activation width {n} is not divisible by 32")
        nbytes = (n // 32) * 34
        buf = (ctypes.c_uint8 * nbytes)()
        rc = self.lib.qwen_quantize_q8_0_scalar(arr, n, buf, nbytes)
        if rc != 0:
            raise RuntimeError(f"Q8_0 activation quantization failed rc={rc}")
        self.activation_quantizations += 1
        return buf, nbytes

    def prepare_activation(self, weight_name: str, x: Sequence[float]):
        arr = self.transform_activation(weight_name, x)
        return self.quantize_q8_0(arr, len(x))

    def matvec_prepared(
        self,
        weights: memoryview,
        meta: Mapping[str, Any],
        prepared,
    ) -> list[float]:
        kind = str(meta["type_name"])
        ne0, rows = map(int, meta["shape"])
        activation, activation_bytes = prepared
        expected_activation_bytes = (ne0 // 32) * 34
        if activation_bytes != expected_activation_bytes:
            raise ValueError(
                f"{meta['name']}: activation bytes={activation_bytes} "
                f"expected={expected_activation_bytes}"
            )
        if kind == "PTQ1_0":
            fn = self.lib.qwen_bonsai2_matvec_ptq1_0_q8_0
        elif kind == "PQ2_0":
            fn = self.lib.qwen_bonsai2_matvec_pq2_0_q8_0
        else:
            raise ValueError(f"unsupported Bonsai 2 native matvec type {kind}")

        w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
        out = (ctypes.c_float * rows)()
        rc = fn(
            w_arr,
            len(weights),
            rows,
            ne0,
            activation,
            activation_bytes,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: native Bonsai 2 matvec failed rc={rc}")
        self.matvec_rows += rows
        return [float(out[i]) for i in range(rows)]

    def matvec(
        self,
        weights: memoryview,
        meta: Mapping[str, Any],
        x: Sequence[float],
    ) -> list[float]:
        ne0 = int(meta["shape"][0])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: input width={len(x)} ne0={ne0}")
        prepared = self.prepare_activation(str(meta["name"]), x)
        return self.matvec_prepared(weights, meta, prepared)

    def report(self) -> dict[str, object]:
        return {
            "backend": "native-c-avx2-k3",
            "hadamard_block_size": self.block_size,
            "hadamard_sign_mode": self.sign_mode,
            "folded_weight_count": len(self.folded_weights),
            "inverse_weight_count": len(self.inverse_weights),
            "gdn_v_grouped": self.gdn_v_grouped,
            "activation_quantizations": self.activation_quantizations,
            "hadamard_transforms": self.hadamard_transforms,
            "matvec_rows": self.matvec_rows,
        }
