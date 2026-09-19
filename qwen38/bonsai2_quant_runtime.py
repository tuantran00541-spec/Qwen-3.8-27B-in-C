#!/usr/bin/env python3
"""Native C runtime adapter for Prism Bonsai 2 low-bit GGUF weights.

This module intentionally exposes only the arithmetic required to validate the
new K3 streaming path first: Hadamard/sign activation preparation, Q8_0
activation quantization, and PTQ1_0/PQ2_0 matvec. Full Qwen3.8 generation is
wired only after these primitives pass real-weight gates.
"""
from __future__ import annotations

from array import array
import ctypes
import time
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
    def __init__(
        self,
        library: Path,
        metadata: Mapping[str, object],
        *,
        threads: int = 1,
        max_rows: int = 300_000,
    ) -> None:
        self.library = Path(library)
        self.lib = ctypes.CDLL(str(self.library))
        self.threads = int(threads)
        if self.threads < 1 or self.threads > 64:
            raise ValueError("threads must be in [1,64]")
        self.pool = None
        self._attention_cache: dict[int, dict[str, object]] = {}

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

        self.lib.qwen_bonsai2_inverse_fwht_blocks.argtypes = [
            _C_FP, ctypes.c_size_t, ctypes.c_size_t, _C_I8P
        ]
        self.lib.qwen_bonsai2_inverse_fwht_blocks.restype = ctypes.c_int
        self.lib.qwen_bonsai2_permute_gdn_ssm_out_f32.argtypes = [
            _C_FP, _C_FP, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.qwen_bonsai2_permute_gdn_ssm_out_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_swiglu_f32.argtypes = [
            _C_FP, _C_FP, ctypes.c_size_t, _C_FP
        ]
        self.lib.qwen_bonsai2_swiglu_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_attention_gate_f32.argtypes = [
            _C_FP, _C_FP, ctypes.c_size_t, _C_FP
        ]
        self.lib.qwen_bonsai2_attention_gate_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_residual_add_f32.argtypes = [
            _C_FP, _C_FP, ctypes.c_size_t, _C_FP
        ]
        self.lib.qwen_bonsai2_residual_add_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_gdn_repeat_scale_f32.argtypes = [
            _C_FP, _C_FP, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, _C_FP, _C_FP
        ]
        self.lib.qwen_bonsai2_gdn_repeat_scale_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_attention_core_f32.argtypes = [
            _C_FP,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _C_FP,
            _C_FP,
            ctypes.c_size_t,
            ctypes.c_double,
            _C_FP,
        ]
        self.lib.qwen_bonsai2_attention_core_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_gdn_conv_silu_f32.argtypes = [
            _C_FP, _C_FP, ctypes.c_size_t, _C_FP, ctypes.c_size_t, _C_FP
        ]
        self.lib.qwen_bonsai2_gdn_conv_silu_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_gdn_norm_gate_f32.argtypes = [
            _C_FP, _C_FP, _C_FP, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, _C_FP
        ]
        self.lib.qwen_bonsai2_gdn_norm_gate_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_recurrent_mid_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            _C_FP,
            _C_FP,
            _C_FP,
            _C_FP,
            _C_FP,
            ctypes.c_size_t,
            _C_FP,
            _C_FP,
            _C_FP,
            _C_FP,
            _C_FP,
            _C_FP,
            _C_FP,
            ctypes.c_float,
            ctypes.c_float,
            _C_FP,
        ]
        self.lib.qwen_bonsai2_recurrent_mid_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_rms_norm_f32.argtypes = [
            _C_FP, _C_FP, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, _C_FP
        ]
        self.lib.qwen_bonsai2_rms_norm_f32.restype = ctypes.c_int
        self.lib.qwen_bonsai2_matvec_bf16_f32.argtypes = [
            _C_U8P, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            _C_FP, _C_FP
        ]
        self.lib.qwen_bonsai2_matvec_bf16_f32.restype = ctypes.c_int
        for name in (
            "qwen_bonsai2_dequantize_ptq1_0_row",
            "qwen_bonsai2_dequantize_pq2_0_row",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [_C_U8P, ctypes.c_size_t, ctypes.c_size_t, _C_FP]
            fn.restype = ctypes.c_int

        if hasattr(self.lib, "qwen_bonsai2_pool_create"):
            self.lib.qwen_bonsai2_pool_create.argtypes = [
                ctypes.c_int, ctypes.c_size_t
            ]
            self.lib.qwen_bonsai2_pool_create.restype = ctypes.c_void_p
            self.lib.qwen_bonsai2_pool_destroy.argtypes = [ctypes.c_void_p]
            self.lib.qwen_bonsai2_pool_destroy.restype = None
            for name in (
                "qwen_bonsai2_pool_matvec_ptq1_0",
                "qwen_bonsai2_pool_matvec_pq2_0",
            ):
                fn = getattr(self.lib, name)
                fn.argtypes = [
                    ctypes.c_void_p,
                    _C_U8P,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    _C_U8P,
                    ctypes.c_size_t,
                    _C_FP,
                ]
                fn.restype = ctypes.c_int
            self.lib.qwen_bonsai2_pool_calls.argtypes = [ctypes.c_void_p]
            self.lib.qwen_bonsai2_pool_calls.restype = ctypes.c_uint64
            self.lib.qwen_bonsai2_pool_threads.argtypes = [ctypes.c_void_p]
            self.lib.qwen_bonsai2_pool_threads.restype = ctypes.c_int
            self.pool = self.lib.qwen_bonsai2_pool_create(
                self.threads, int(max_rows)
            )
            if not self.pool:
                raise RuntimeError(
                    f"failed to create Bonsai 2 persistent pool threads={self.threads}"
                )
        elif self.threads != 1:
            raise RuntimeError(
                "native library has no Bonsai 2 persistent pool but threads > 1"
            )

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
        timing_keys = (
            "activation_pack",
            "hadamard",
            "q8_quantize",
            "ptq1_matvec",
            "pq2_matvec",
            "bf16_matvec",
            "lookup_dequantize",
            "swiglu",
            "attention_gate",
            "attention_core",
            "residual_add",
            "gdn_conv_silu",
            "gdn_repeat_scale",
            "gdn_norm_gate",
            "recurrent_mid",
            "rms_norm",
            "output_copy",
        )
        self.timing_seconds = {key: 0.0 for key in timing_keys}
        self.timing_calls = {key: 0 for key in timing_keys}

    def _record_timing(self, key: str, started: float) -> None:
        self.timing_seconds[key] += time.perf_counter() - started
        self.timing_calls[key] += 1

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
        started = time.perf_counter()
        arr = (ctypes.c_float * n)(*map(float, x))
        self._record_timing("activation_pack", started)
        if weight_name not in self.folded_weights:
            return arr

        started = time.perf_counter()
        if self.gdn_v_grouped and ".ssm_out." in weight_name:
            if n % (48 * 128) != 0 or n != 48 * 128:
                raise ValueError(f"{weight_name}: unexpected Qwen3.5 GDN width {n}")
            grouped = (ctypes.c_float * n)()
            rc = self.lib.qwen_bonsai2_permute_gdn_ssm_out_f32(
                arr, grouped, n, 128, 16, 3
            )
            if rc != 0:
                raise RuntimeError(
                    f"{weight_name}: native GDN grouped permutation failed rc={rc}"
                )
            arr = grouped

        sign_owner, sign_ptr = self._sign_array(n)
        rc = self.lib.qwen_bonsai2_fwht_blocks(arr, n, self.block_size, sign_ptr)
        _ = sign_owner
        if rc != 0:
            raise RuntimeError(f"{weight_name}: native Hadamard failed rc={rc}")
        self.hadamard_transforms += 1
        self._record_timing("hadamard", started)
        return arr

    def quantize_q8_0(self, arr, n: int):
        if n % 32:
            raise ValueError(f"Q8_0 activation width {n} is not divisible by 32")
        nbytes = (n // 32) * 34
        buf = (ctypes.c_uint8 * nbytes)()
        started = time.perf_counter()
        rc = self.lib.qwen_quantize_q8_0_scalar(arr, n, buf, nbytes)
        self._record_timing("q8_quantize", started)
        if rc != 0:
            raise RuntimeError(f"Q8_0 activation quantization failed rc={rc}")
        self.activation_quantizations += 1
        return buf, nbytes

    def prepare_activation(self, weight_name: str, x: Sequence[float]):
        arr = self.transform_activation(weight_name, x)
        return self.quantize_q8_0(arr, len(x))

    def swiglu(
        self,
        gate: Sequence[float],
        up: Sequence[float],
    ) -> list[float]:
        if len(gate) != len(up):
            raise ValueError(
                f"SwiGLU shape mismatch gate={len(gate)} up={len(up)}"
            )
        n = len(gate)
        if n == 0:
            return []
        gate_arr = (ctypes.c_float * n)(*map(float, gate))
        up_arr = (ctypes.c_float * n)(*map(float, up))
        out = (ctypes.c_float * n)()
        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_swiglu_f32(
            gate_arr, up_arr, n, out
        )
        self._record_timing("swiglu", started)
        if rc != 0:
            raise RuntimeError(f"native Bonsai 2 SwiGLU failed rc={rc}")
        started = time.perf_counter()
        result = self._marshal_f32_output(out, n)
        self._record_timing("output_copy", started)
        return result

    def attention_sigmoid_mul(
        self,
        pregate: Sequence[float],
        gate: Sequence[float],
    ) -> list[float]:
        if len(pregate) != len(gate):
            raise ValueError(
                f"attention gate shape mismatch pregate={len(pregate)} gate={len(gate)}"
            )
        n = len(pregate)
        if n == 0:
            return []
        pregate_arr = (ctypes.c_float * n)(*map(float, pregate))
        gate_arr = (ctypes.c_float * n)(*map(float, gate))
        out = (ctypes.c_float * n)()
        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_attention_gate_f32(
            pregate_arr, gate_arr, n, out
        )
        self._record_timing("attention_gate", started)
        if rc != 0:
            raise RuntimeError(f"native Bonsai 2 attention gate failed rc={rc}")
        started = time.perf_counter()
        result = self._marshal_f32_output(out, n)
        self._record_timing("output_copy", started)
        return result

    @staticmethod
    def _array_f32_ptr(buf: array):
        return (ctypes.c_float * len(buf)).from_buffer(buf)

    @staticmethod
    def _borrow_f32(values, expected: int):
        expected = int(expected)
        if expected < 0:
            raise ValueError("expected F32 length must be non-negative")
        if isinstance(values, ctypes.Array):
            if len(values) != expected:
                raise ValueError(
                    f"F32 ctypes length={len(values)} expected={expected}"
                )
            return values, ctypes.cast(values, _C_FP)
        if isinstance(values, array):
            if values.typecode != "f" or len(values) != expected:
                raise ValueError(
                    f"F32 array length={len(values)} expected={expected}"
                )
            owner = (ctypes.c_float * expected).from_buffer(values)
            return owner, ctypes.cast(owner, _C_FP)
        if isinstance(values, memoryview):
            if values.nbytes != expected * ctypes.sizeof(ctypes.c_float):
                raise ValueError(
                    f"F32 view bytes={values.nbytes} expected={expected * 4}"
                )
            if values.readonly:
                copied = array("f")
                copied.frombytes(values.cast("B").tobytes())
                owner = (ctypes.c_float * expected).from_buffer(copied)
                return (copied, owner), ctypes.cast(owner, _C_FP)
            owner = (ctypes.c_float * expected).from_buffer(values)
            return owner, ctypes.cast(owner, _C_FP)

        copied = array("f", map(float, values))
        if len(copied) != expected:
            raise ValueError(
                f"F32 sequence length={len(copied)} expected={expected}"
            )
        owner = (ctypes.c_float * expected).from_buffer(copied)
        return (copied, owner), ctypes.cast(owner, _C_FP)

    @staticmethod
    def _marshal_f32_output(out, n: int) -> list[float]:
        n = int(n)
        if n < 0 or len(out) != n:
            raise ValueError(f"F32 output length={len(out)} expected={n}")
        if n == 0:
            return []
        flat = array("f")
        flat.frombytes(memoryview(out).cast("B"))
        if len(flat) != n:
            raise RuntimeError(
                f"bulk F32 marshal length={len(flat)} expected={n}"
            )
        return flat.tolist()

    def _sync_attention_cache(
        self,
        layer: int,
        cache,
        kv_dim: int,
    ) -> tuple[array, array, int]:
        n_ctx = len(cache["k"])
        if n_ctx <= 0 or n_ctx != len(cache["v"]):
            raise ValueError("attention K/V cache length mismatch")
        newest_k = cache["k"][-1]
        newest_v = cache["v"][-1]
        if len(newest_k) != kv_dim or len(newest_v) != kv_dim:
            raise ValueError("attention K/V cache width mismatch")

        slot = self._attention_cache.get(int(layer))
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
            self._attention_cache[int(layer)] = slot
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

    def attention_core(
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
        q_heads = int(q_heads)
        kv_heads = int(kv_heads)
        head_dim = int(head_dim)
        if q_heads <= 0 or kv_heads <= 0 or head_dim <= 0 or q_heads % kv_heads:
            raise ValueError(
                f"attention geometry q_heads={q_heads} kv_heads={kv_heads} "
                f"head_dim={head_dim}"
            )
        q_dim = q_heads * head_dim
        kv_dim = kv_heads * head_dim
        if len(q) != q_dim:
            raise ValueError(f"attention Q width={len(q)} expected={q_dim}")

        qbuf = array("f", map(float, q))
        kflat, vflat, n_ctx = self._sync_attention_cache(
            int(layer), cache, kv_dim
        )
        out = (ctypes.c_float * q_dim)()
        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_attention_core_f32(
            self._array_f32_ptr(qbuf),
            q_heads,
            kv_heads,
            head_dim,
            self._array_f32_ptr(kflat),
            self._array_f32_ptr(vflat),
            n_ctx,
            ctypes.c_double(float(scale)),
            out,
        )
        self._record_timing("attention_core", started)
        if rc != 0:
            raise RuntimeError(f"native Bonsai 2 attention core failed rc={rc}")

        started = time.perf_counter()
        result = self._marshal_f32_output(out, q_dim)
        self._record_timing("output_copy", started)
        return result

    def gdn_repeat_scale(
        self,
        q: Sequence[float],
        k: Sequence[float],
        *,
        repeats: int,
        scale: float,
    ) -> tuple[list[float], list[float]]:
        key_dim = len(q)
        repeats = int(repeats)
        if key_dim == 0 or len(k) != key_dim or repeats <= 0:
            raise ValueError(
                f"GDN repeat-scale shape mismatch q={len(q)} k={len(k)} "
                f"repeats={repeats}"
            )
        q_arr = array("f", map(float, q))
        k_arr = array("f", map(float, k))
        out_width = key_dim * repeats
        q_out = array("f", [0.0]) * out_width
        k_out = array("f", [0.0]) * out_width

        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_gdn_repeat_scale_f32(
            self._array_f32_ptr(q_arr),
            self._array_f32_ptr(k_arr),
            key_dim,
            repeats,
            ctypes.c_float(float(scale)),
            self._array_f32_ptr(q_out),
            self._array_f32_ptr(k_out),
        )
        self._record_timing("gdn_repeat_scale", started)
        if rc != 0:
            raise RuntimeError(f"native GDN repeat-scale failed rc={rc}")

        started = time.perf_counter()
        result = (q_out.tolist(), k_out.tolist())
        self._record_timing("output_copy", started)
        return result

    def residual_add(
        self,
        a: Sequence[float],
        b: Sequence[float],
    ) -> list[float]:
        if len(a) != len(b):
            raise ValueError(
                f"residual-add shape mismatch a={len(a)} b={len(b)}"
            )
        n = len(a)
        if n == 0:
            return []
        a_arr = (ctypes.c_float * n)(*map(float, a))
        b_arr = (ctypes.c_float * n)(*map(float, b))
        out = (ctypes.c_float * n)()
        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_residual_add_f32(a_arr, b_arr, n, out)
        self._record_timing("residual_add", started)
        if rc != 0:
            raise RuntimeError(f"native Bonsai 2 residual add failed rc={rc}")
        started = time.perf_counter()
        result = self._marshal_f32_output(out, n)
        self._record_timing("output_copy", started)
        return result

    def gdn_conv_silu(
        self,
        qkv: Sequence[float],
        history: Sequence[Sequence[float]],
        kernels: Sequence[float],
    ) -> list[float]:
        n = len(qkv)
        if n == 0:
            return []
        if len(kernels) != n * 4:
            raise ValueError(
                f"GDN conv kernel shape mismatch qkv={n} kernels={len(kernels)}"
            )
        selected = list(history[-3:])
        for row in selected:
            if len(row) != n:
                raise ValueError(
                    f"GDN conv history shape mismatch qkv={n} history={len(row)}"
                )

        qkv_arr = (ctypes.c_float * n)(*map(float, qkv))
        kernel_arr = (ctypes.c_float * (n * 4))(*map(float, kernels))
        history_count = len(selected)
        if history_count:
            history_arr = (ctypes.c_float * (history_count * n))(
                *(float(v) for row in selected for v in row)
            )
            history_ptr = ctypes.cast(history_arr, _C_FP)
        else:
            history_arr = None
            history_ptr = None
        out = (ctypes.c_float * n)()

        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_gdn_conv_silu_f32(
            qkv_arr,
            history_ptr,
            history_count,
            kernel_arr,
            n,
            out,
        )
        self._record_timing("gdn_conv_silu", started)
        _ = history_arr
        if rc != 0:
            raise RuntimeError(f"native GDN conv+SiLU failed rc={rc}")

        started = time.perf_counter()
        result = self._marshal_f32_output(out, n)
        self._record_timing("output_copy", started)
        return result

    def recurrent_mid(
        self,
        state_lib,
        state,
        qkv,
        history,
        kernels,
        alpha,
        beta_raw,
        dt,
        a,
        z,
        norm_weight,
        *,
        eps: float,
        scale: float,
    ) -> list[float]:
        key_heads = 16
        value_heads = 48
        head_dim = 128
        key_dim = key_heads * head_dim
        value_dim = value_heads * head_dim
        conv_dim = 2 * key_dim + value_dim
        state_elems = value_heads * head_dim * head_dim

        if len(history) > 3:
            raise ValueError(f"GDN history rows={len(history)} exceeds 3")

        owners = []
        for values, expected in (
            (state, state_elems),
            (qkv, conv_dim),
            (kernels, conv_dim * 4),
            (alpha, value_heads),
            (beta_raw, value_heads),
            (dt, value_heads),
            (a, value_heads),
            (z, value_dim),
            (norm_weight, head_dim),
        ):
            owner, _ = self._borrow_f32(values, expected)
            owners.append(owner)

        state_owner, state_ptr = self._borrow_f32(state, state_elems)
        qkv_owner, qkv_ptr = self._borrow_f32(qkv, conv_dim)
        kernels_owner, kernels_ptr = self._borrow_f32(kernels, conv_dim * 4)
        alpha_owner, alpha_ptr = self._borrow_f32(alpha, value_heads)
        beta_owner, beta_ptr = self._borrow_f32(beta_raw, value_heads)
        dt_owner, dt_ptr = self._borrow_f32(dt, value_heads)
        a_owner, a_ptr = self._borrow_f32(a, value_heads)
        z_owner, z_ptr = self._borrow_f32(z, value_dim)
        norm_owner, norm_ptr = self._borrow_f32(norm_weight, head_dim)

        history_owners = []
        history_ptrs = [None, None, None]
        for i, row in enumerate(history):
            owner, row_ptr = self._borrow_f32(row, conv_dim)
            history_owners.append(owner)
            history_ptrs[i] = row_ptr

        out = (ctypes.c_float * value_dim)()
        step_fn = state_lib.lib.qwen_gdn_pool_step_f32
        step_ptr = ctypes.cast(step_fn, ctypes.c_void_p)

        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_recurrent_mid_f32(
            state_lib.pool,
            step_ptr,
            state_ptr,
            qkv_ptr,
            history_ptrs[0],
            history_ptrs[1],
            history_ptrs[2],
            len(history),
            kernels_ptr,
            alpha_ptr,
            beta_ptr,
            dt_ptr,
            a_ptr,
            z_ptr,
            norm_ptr,
            ctypes.c_float(float(eps)),
            ctypes.c_float(float(scale)),
            out,
        )
        self._record_timing("recurrent_mid", started)
        _ = (
            owners,
            state_owner,
            qkv_owner,
            kernels_owner,
            alpha_owner,
            beta_owner,
            dt_owner,
            a_owner,
            z_owner,
            norm_owner,
            history_owners,
        )
        if rc != 0:
            raise RuntimeError(f"native recurrent mid failed rc={rc}")

        started = time.perf_counter()
        result = self._marshal_f32_output(out, value_dim)
        self._record_timing("output_copy", started)
        return result

    def gdn_norm_gate(
        self,
        core: Sequence[float],
        norm_weight: Sequence[float],
        z: Sequence[float],
        *,
        eps: float = 1e-6,
    ) -> list[float]:
        n = len(core)
        head_dim = len(norm_weight)
        if n == 0:
            return []
        if len(z) != n or head_dim == 0 or n % head_dim:
            raise ValueError(
                f"GDN norm-gate shape mismatch core={n} z={len(z)} "
                f"head_dim={head_dim}"
            )
        heads = n // head_dim
        core_arr = (ctypes.c_float * n)(*map(float, core))
        weight_arr = (ctypes.c_float * head_dim)(*map(float, norm_weight))
        z_arr = (ctypes.c_float * n)(*map(float, z))
        out = (ctypes.c_float * n)()

        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_gdn_norm_gate_f32(
            core_arr,
            weight_arr,
            z_arr,
            heads,
            head_dim,
            ctypes.c_float(float(eps)),
            out,
        )
        self._record_timing("gdn_norm_gate", started)
        if rc != 0:
            raise RuntimeError(f"native GDN norm+gate failed rc={rc}")

        started = time.perf_counter()
        result = self._marshal_f32_output(out, n)
        self._record_timing("output_copy", started)
        return result

    def rms_norm(
        self,
        values: Sequence[float],
        weight: Sequence[float],
        *,
        rows: int = 1,
        eps: float = 1e-6,
    ) -> list[float]:
        n = len(values)
        width = len(weight)
        if rows < 1 or width < 1 or n != rows * width:
            raise ValueError(
                f"RMSNorm shape mismatch values={n} rows={rows} width={width}"
            )
        x_arr = (ctypes.c_float * n)(*map(float, values))
        w_arr = (ctypes.c_float * width)(*map(float, weight))
        out = (ctypes.c_float * n)()
        started = time.perf_counter()
        rc = self.lib.qwen_bonsai2_rms_norm_f32(
            x_arr, w_arr, rows, width, ctypes.c_float(float(eps)), out
        )
        self._record_timing("rms_norm", started)
        if rc != 0:
            raise RuntimeError(f"native RMSNorm failed rc={rc}")
        started = time.perf_counter()
        result = self._marshal_f32_output(out, n)
        self._record_timing("output_copy", started)
        return result

    def matvec_prepared(
        self,
        weights: memoryview,
        meta: Mapping[str, Any],
        prepared,
    ) -> list[float]:
        kind = str(meta["type_name"])
        ne0, rows = map(int, meta["shape"])
        if kind == "BF16":
            activation = prepared
            if not isinstance(activation, ctypes.Array):
                activation = (ctypes.c_float * ne0)(*map(float, activation))
            w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
            out = (ctypes.c_float * rows)()
            started = time.perf_counter()
            rc = self.lib.qwen_bonsai2_matvec_bf16_f32(
                w_arr, len(weights), rows, ne0, activation, out
            )
            self._record_timing("bf16_matvec", started)
            if rc != 0:
                raise RuntimeError(f"{meta['name']}: native BF16 matvec failed rc={rc}")
            self.matvec_rows += rows
            started = time.perf_counter()
            result = self._marshal_f32_output(out, rows)
            self._record_timing("output_copy", started)
            return result

        activation, activation_bytes = prepared
        expected_activation_bytes = (ne0 // 32) * 34
        if activation_bytes != expected_activation_bytes:
            raise ValueError(
                f"{meta['name']}: activation bytes={activation_bytes} "
                f"expected={expected_activation_bytes}"
            )
        if kind == "PTQ1_0":
            fn = (
                self.lib.qwen_bonsai2_pool_matvec_ptq1_0
                if self.pool
                else self.lib.qwen_bonsai2_matvec_ptq1_0_q8_0
            )
        elif kind == "PQ2_0":
            fn = (
                self.lib.qwen_bonsai2_pool_matvec_pq2_0
                if self.pool
                else self.lib.qwen_bonsai2_matvec_pq2_0_q8_0
            )
        else:
            raise ValueError(f"unsupported Bonsai 2 native matvec type {kind}")

        w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
        out = (ctypes.c_float * rows)()
        common = (
            w_arr,
            len(weights),
            rows,
            ne0,
            activation,
            activation_bytes,
            out,
        )
        timing_key = "ptq1_matvec" if kind == "PTQ1_0" else "pq2_matvec"
        started = time.perf_counter()
        rc = fn(self.pool, *common) if self.pool else fn(*common)
        self._record_timing(timing_key, started)
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: native Bonsai 2 matvec failed rc={rc}")
        self.matvec_rows += rows
        started = time.perf_counter()
        result = self._marshal_f32_output(out, rows)
        self._record_timing("output_copy", started)
        return result

    def matvec(
        self,
        weights: memoryview,
        meta: Mapping[str, Any],
        x: Sequence[float],
    ) -> list[float]:
        ne0 = int(meta["shape"][0])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: input width={len(x)} ne0={ne0}")
        if str(meta["type_name"]) == "BF16":
            prepared = (ctypes.c_float * ne0)(*map(float, x))
        else:
            prepared = self.prepare_activation(str(meta["name"]), x)
        return self.matvec_prepared(weights, meta, prepared)

    def dequantize_lookup_row(
        self,
        raw: bytes | bytearray | memoryview,
        kind: str,
        n: int,
        weight_name: str,
    ) -> list[float]:
        if kind == "PTQ1_0":
            fn = self.lib.qwen_bonsai2_dequantize_ptq1_0_row
        elif kind == "PQ2_0":
            fn = self.lib.qwen_bonsai2_dequantize_pq2_0_row
        else:
            raise ValueError(f"unsupported Bonsai 2 lookup type {kind}")
        buf = bytearray(raw)
        src = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
        out = (ctypes.c_float * n)()
        started = time.perf_counter()
        rc = fn(src, len(buf), n, out)
        if rc != 0:
            raise RuntimeError(f"{weight_name}: dequantize row failed rc={rc}")

        if weight_name in self.inverse_weights:
            sign_owner, sign_ptr = self._sign_array(n)
            rc = self.lib.qwen_bonsai2_inverse_fwht_blocks(
                out, n, self.block_size, sign_ptr
            )
            _ = sign_owner
            if rc != 0:
                raise RuntimeError(
                    f"{weight_name}: inverse Hadamard lookup failed rc={rc}"
                )
        self._record_timing("lookup_dequantize", started)
        started = time.perf_counter()
        result = self._marshal_f32_output(out, n)
        self._record_timing("output_copy", started)
        return result

    def report(self) -> dict[str, object]:
        return {
            "backend": "native-c-avx2-k3",
            "threads": self.threads,
            "persistent_pool": bool(self.pool),
            "pool_calls": (
                int(self.lib.qwen_bonsai2_pool_calls(self.pool))
                if self.pool else 0
            ),
            "hadamard_block_size": self.block_size,
            "hadamard_sign_mode": self.sign_mode,
            "folded_weight_count": len(self.folded_weights),
            "inverse_weight_count": len(self.inverse_weights),
            "gdn_v_grouped": self.gdn_v_grouped,
            "activation_quantizations": self.activation_quantizations,
            "hadamard_transforms": self.hadamard_transforms,
            "matvec_rows": self.matvec_rows,
            "timing_seconds": dict(self.timing_seconds),
            "timing_calls": dict(self.timing_calls),
        }


    def close(self) -> None:
        self._attention_cache.clear()
        if self.pool:
            self.lib.qwen_bonsai2_pool_destroy(self.pool)
            self.pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
