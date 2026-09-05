#!/usr/bin/env python3
"""Windows ABI/import gate for the complete exact Qwen3.8 native helper stack.

This gate is intentionally model-free.  It proves that the already-validated C
sources can be built as Windows DLLs with the exact symbols expected by Python,
and that the full generator import graph survives the Windows platform shims.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

from qwen38_win32_bootstrap import import_generator


DLL_SYMBOLS = {
    "qwen_quant_base.dll": (
        "qwen_quantize_q8_k_scalar",
        "qwen_quantize_q8_0_scalar",
        "qwen_matvec_q6_k_q8_k_scalar",
        "qwen_matvec_q8_0_q8_0_scalar",
        "qwen_matvec_many_q6_k_q8_k_bridge",
        "qwen_matvec_many_q8_0_q8_0_bridge",
    ),
    "qwen_q6_portable.dll": (
        "qwen_q6_pool_create",
        "qwen_q6_pool_destroy",
        "qwen_q6_pool_matvec_many",
        "qwen_q6_pool_calls",
        "qwen_q6_pool_threads",
        "qwen_matvec_many_q6_k_q8_k_bridge",
        "qwen_matvec_many_q8_0_q8_0_bridge",
    ),
    "qwen_gdn_state.dll": ("qwen_gdn_ar_step_f32",),
    "qwen_gdn_state_batch.dll": ("qwen_gdn_ar_batch_f32",),
    "qwen_f32.dll": ("qwen_matvec_f32_fsum_exact",),
    "qwen_attention_core.dll": ("qwen_attention_core_f32_exact",),
    "qwen_gdn_conv_silu.dll": ("qwen_gdn_conv_silu_many_f32_exact",),
    "qwen_gdn_output_gate.dll": ("qwen_gdn_output_rmsnorm_gate_f32_exact",),
    "qwen_swiglu.dll": ("qwen_swiglu_many_f32_exact",),
    "qwen_rmsnorm.dll": (
        "qwen38_rmsnorm_exact_f32",
        "qwen38_rmsnorm_heads_exact_f32",
    ),
    "qwen_rmsnorm_many.dll": ("qwen38_rmsnorm_many_exact_f32",),
    "qwen_rmsnorm_heads_many.dll": ("qwen38_rmsnorm_heads_many_exact_f32",),
    "qwen_residual_add.dll": ("qwen38_residual_add_many_exact_f32",),
    "qwen_attention_gate.dll": ("qwen38_attention_gate_exact_f32",),
    "qwen_gdn_repeat_scale.dll": ("qwen38_gdn_repeat_scale_many_exact_f32",),
    "qwen_win32_direct_io.dll": (
        "qwen_win32_direct_open_utf8",
        "qwen_win32_direct_close",
        "qwen_win32_direct_alignment",
        "qwen_win32_direct_read",
        "qwen_win32_direct_no_buffering",
        "qwen_win32_direct_overlapped",
        "qwen_win32_buffer_create",
        "qwen_win32_buffer_ptr",
        "qwen_win32_buffer_destroy",
    ),
}


def _load_symbols(dll_dir: Path):
    loaded = {}
    for filename, symbols in DLL_SYMBOLS.items():
        path = dll_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        lib = ctypes.CDLL(str(path))
        for symbol in symbols:
            getattr(lib, symbol)
        loaded[filename] = {"path": str(path), "symbols": list(symbols)}
    return loaded


def _python_wrapper_gate(dll_dir: Path) -> dict:
    # Exercise the loaders used by the actual decoder, not only bare getattr().
    import qwen35_gdn_quant_layer_gate as gdn
    import qwen35_k3_two_token as two
    from native_f32_runtime import load_f32_lib
    from attention_core_runtime import ExactAttentionCore
    from gdn_conv_silu_runtime import ExactGDNConvSilu
    from gdn_output_gate_runtime import ExactGDNOutputGate
    from swiglu_runtime import ExactSwiGLU
    from rmsnorm_runtime import ExactRMSNorm
    from rmsnorm_many_runtime import ExactRMSNormMany
    from rmsnorm_heads_many_runtime import ExactRMSNormHeadsMany
    from residual_add_runtime import ExactResidualAdd
    from attention_gate_runtime import ExactAttentionGate
    from gdn_repeat_scale_runtime import ExactGDNRepeatScale
    from gdn_state_batch_runtime import ExactGDNStateBatch
    from qwen38_current_best_runtime_win32 import (
        Qwen38CurrentBestWin32QuantStack,
        sanity as quant_sanity,
    )

    gdn._load_native(dll_dir / "qwen_quant_base.dll")
    two.load_state_lib(dll_dir / "qwen_gdn_state.dll")
    load_f32_lib(dll_dir / "qwen_f32.dll")
    ExactAttentionCore(dll_dir / "qwen_attention_core.dll")
    ExactGDNConvSilu(dll_dir / "qwen_gdn_conv_silu.dll")
    ExactGDNOutputGate(dll_dir / "qwen_gdn_output_gate.dll")
    ExactSwiGLU(dll_dir / "qwen_swiglu.dll")
    ExactRMSNorm(dll_dir / "qwen_rmsnorm.dll")
    ExactRMSNormMany(dll_dir / "qwen_rmsnorm_many.dll")
    ExactRMSNormHeadsMany(dll_dir / "qwen_rmsnorm_heads_many.dll")
    ExactResidualAdd(dll_dir / "qwen_residual_add.dll")
    ExactAttentionGate(dll_dir / "qwen_attention_gate.dll")
    ExactGDNRepeatScale(dll_dir / "qwen_gdn_repeat_scale.dll")
    ExactGDNStateBatch(dll_dir / "qwen_gdn_state_batch.dll")
    quant_sanity()

    # Keep the composition gate model-free while verifying the real pool ABI.
    class DummyRuntime:
        def __init__(self):
            self.lib = ctypes.CDLL(str(dll_dir / "qwen_q6_portable.dll"))
            self.activation_quantizations = 0
            self.matvec_rows = 0

        def quantize(self, x, kind):
            raise RuntimeError("model-free ABI gate must not execute quantization")

        def matvec(self, weights, meta, x, prepared=None):
            raise RuntimeError("model-free ABI gate must not execute matvec")

    class DummyEngine:
        def __init__(self):
            self.runtime = DummyRuntime()

    engine = DummyEngine()
    base = engine.runtime
    stack = Qwen38CurrentBestWin32QuantStack(
        engine, dll_dir / "qwen_q6_portable.dll"
    )
    report = stack.report()
    if report["platform"] != "win32" or int(report["q6_workers"]) != 2:
        raise RuntimeError(f"unexpected Win32 current-best report: {report}")
    if int(report["q6_pool"]["threads"]) != 2:
        raise RuntimeError(f"unexpected Q6 pool report: {report}")
    if not report["q8_noalloc"] or not report["q6_static_disjoint_rows"]:
        raise RuntimeError(f"exact quant contract changed: {report}")
    if report["arithmetic_change"]:
        raise RuntimeError(f"arithmetic-change flag unexpectedly true: {report}")
    stack.close()
    if engine.runtime is not base:
        raise RuntimeError("Win32 quant stack did not restore engine runtime")

    return {
        "quant_loader": True,
        "state_loader": True,
        "f32_loader": True,
        "exact_helper_wrappers": True,
        "current_best_quant_composition": True,
        "current_best_quant_report": report,
    }


def run(dll_dir: Path) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("Win32 native-stack ABI gate must run on Windows")
    generator = import_generator()
    if int(generator.N_LAYER) != 64:
        raise RuntimeError(f"unexpected generator layer count {generator.N_LAYER}")
    if generator.exact._LIBM is None or float(generator.exact.expf(0.0)) != 1.0:
        raise RuntimeError("UCRT expf binding is not active")

    loaded = _load_symbols(dll_dir)
    wrappers = _python_wrapper_gate(dll_dir)
    return {
        "schema": "qwen38-win32-native-stack-abi-v1",
        "status": "PASS",
        "platform": sys.platform,
        "generator_import": True,
        "generator_layers": int(generator.N_LAYER),
        "ucrt_expf_bound": True,
        "dll_count": len(loaded),
        "dlls": loaded,
        "python_wrappers": wrappers,
        "model_loaded": False,
        "arithmetic_change": False,
        "linux_runtime_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.dll_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("QWEN38_WIN32_NATIVE_STACK_ABI_PASS")


if __name__ == "__main__":
    main()
