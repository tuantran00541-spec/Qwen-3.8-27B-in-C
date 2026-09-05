#!/usr/bin/env python3
"""Windows-only bootstrap for the exact Qwen3.8 runtime import graph.

Legacy evidence modules intentionally keep their proven Linux code unchanged.
On Windows, Python lacks the ``resource`` module, although these modules use it
only for peak-RSS diagnostics.  This bootstrap installs a minimal compatibility
module before importing the generator.  By default it binds UCRT ``expf``;
when ``QWEN38_EXPF_COMPAT_LIB`` is set, the exact Windows gate instead binds
the audited glibc-compatible ``expf`` shim so Linux hidden/state anchors remain
meaningful across platforms.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import struct
import sys
import types


def _peak_working_set_kib() -> int:
    if sys.platform != "win32":
        raise RuntimeError("Win32 peak working set requested on non-Windows host")

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    # Linux resource.ru_maxrss is KiB; preserve that unit so the existing
    # evidence helpers continue to divide by 1024**2 to report GiB.
    return int(counters.PeakWorkingSetSize) // 1024


def install_resource_compat():
    """Return native ``resource`` or install the minimal Win32 diagnostic shim."""
    try:
        return importlib.import_module("resource")
    except ImportError:
        if sys.platform != "win32":
            raise

    existing = sys.modules.get("resource")
    if existing is not None:
        return existing

    module = types.ModuleType("resource")
    module.RUSAGE_SELF = 0

    def getrusage(who: int):
        if int(who) != int(module.RUSAGE_SELF):
            raise ValueError("Win32 resource compatibility supports RUSAGE_SELF only")
        return types.SimpleNamespace(ru_maxrss=_peak_working_set_kib())

    module.getrusage = getrusage
    sys.modules["resource"] = module
    return module


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _bind_ucrt_expf(exact_module):
    """Bind the existing exact wrapper to Windows UCRT ``expf``."""
    if sys.platform != "win32":
        return getattr(exact_module, "_LIBM", None)
    ucrt = ctypes.CDLL("ucrtbase.dll")
    ucrt.expf.argtypes = [ctypes.c_float]
    ucrt.expf.restype = ctypes.c_float
    exact_module._LIBM = ucrt
    return ucrt


def _bind_compat_expf(exact_module, path: str):
    """Bind the audited Win32 shim that reproduces pinned glibc expf values."""
    if sys.platform != "win32":
        raise RuntimeError("Win32 expf compatibility binding requires native Windows")
    if not os.path.isfile(path):
        raise RuntimeError(f"Win32 expf compatibility DLL missing: {path}")
    lib = ctypes.CDLL(path)
    fn = lib.qwen38_glibc_expf_compat
    fn.argtypes = [ctypes.c_float]
    fn.restype = ctypes.c_float
    backend = types.SimpleNamespace(expf=fn, _library=lib, _path=path)
    exact_module._LIBM = backend

    # These are pinned Linux/glibc F32 results from the cross-platform probe.
    # They catch accidental fallback to UCRT before the expensive GGUF gate.
    cases = (
        (-25.2421875, 0x2D3FC4B7),
        (2.5060043334960938, 0x41441803),
        (8.7109375, 0x45BDA770),
    )
    for value, expected in cases:
        actual = _f32_bits(exact_module.expf(value))
        if actual != expected:
            raise RuntimeError(
                f"Win32 expf compatibility mismatch x={value}: "
                f"actual=0x{actual:08x} expected=0x{expected:08x}"
            )
    return backend


def _bind_win32_expf(exact_module):
    """Select UCRT or the explicitly requested exact compatibility backend."""
    if sys.platform != "win32":
        return getattr(exact_module, "_LIBM", None)
    compat = os.environ.get("QWEN38_EXPF_COMPAT_LIB", "").strip()
    if compat:
        return _bind_compat_expf(exact_module, compat)
    return _bind_ucrt_expf(exact_module)


def import_generator():
    """Import the proven generator graph without editing its legacy modules."""
    install_resource_compat()
    generator = importlib.import_module("qwen35_k3_generate")
    _bind_win32_expf(generator.exact)
    return generator


def sanity() -> None:
    if sys.platform != "win32":
        raise SystemExit("Win32 bootstrap sanity must run on Windows")
    resource = install_resource_compat()
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss <= 0:
        raise SystemExit(f"invalid Win32 peak RSS: {rss}")
    generator = import_generator()
    if int(generator.N_LAYER) != 64:
        raise SystemExit(f"unexpected generator layer count: {generator.N_LAYER}")
    if generator.exact._LIBM is None:
        raise SystemExit("Windows expf backend was not bound")
    if float(generator.exact.expf(0.0)) != 1.0:
        raise SystemExit("Windows expf backend sanity failed")
    if os.environ.get("QWEN38_EXPF_COMPAT_LIB", "").strip():
        print("QWEN38_WIN32_GLIBC_EXPF_COMPAT_BIND_PASS")
    print("QWEN38_WIN32_GENERATOR_BOOTSTRAP_PASS")


if __name__ == "__main__":
    sanity()
