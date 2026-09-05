#!/usr/bin/env python3
"""ctypes wrapper and synthetic gate for native Win32 unbuffered K3 reads."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Any


class Win32DirectIO:
    def __init__(self, library: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Win32DirectIO requires Windows")
        self.lib = ctypes.CDLL(str(library))
        c_void_p = ctypes.c_void_p
        c_u32 = ctypes.c_uint32
        c_u64 = ctypes.c_uint64

        self.lib.qwen_win32_direct_open_utf8.argtypes = [ctypes.c_char_p]
        self.lib.qwen_win32_direct_open_utf8.restype = c_void_p
        self.lib.qwen_win32_direct_close.argtypes = [c_void_p]
        self.lib.qwen_win32_direct_close.restype = ctypes.c_int

        for name in (
            "qwen_win32_direct_logical_sector",
            "qwen_win32_direct_physical_sector",
            "qwen_win32_direct_alignment",
            "qwen_win32_direct_last_error",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [c_void_p]
            fn.restype = c_u32

        for name in ("qwen_win32_direct_calls", "qwen_win32_direct_bytes"):
            fn = getattr(self.lib, name)
            fn.argtypes = [c_void_p]
            fn.restype = c_u64

        self.lib.qwen_win32_direct_seconds.argtypes = [c_void_p]
        self.lib.qwen_win32_direct_seconds.restype = ctypes.c_double
        for name in (
            "qwen_win32_direct_no_buffering",
            "qwen_win32_direct_overlapped",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [c_void_p]
            fn.restype = ctypes.c_int

        self.lib.qwen_win32_buffer_create.argtypes = [c_u64, c_u32]
        self.lib.qwen_win32_buffer_create.restype = c_void_p
        self.lib.qwen_win32_buffer_ptr.argtypes = [c_void_p]
        self.lib.qwen_win32_buffer_ptr.restype = c_void_p
        self.lib.qwen_win32_buffer_bytes.argtypes = [c_void_p]
        self.lib.qwen_win32_buffer_bytes.restype = c_u64
        self.lib.qwen_win32_buffer_alignment.argtypes = [c_void_p]
        self.lib.qwen_win32_buffer_alignment.restype = c_u32
        self.lib.qwen_win32_buffer_destroy.argtypes = [c_void_p]
        self.lib.qwen_win32_buffer_destroy.restype = ctypes.c_int

        self.lib.qwen_win32_direct_read.argtypes = [
            c_void_p, c_void_p, c_u64, c_u64, c_u32,
        ]
        self.lib.qwen_win32_direct_read.restype = ctypes.c_int

    def open(self, path: Path) -> int:
        handle = self.lib.qwen_win32_direct_open_utf8(
            os.fsencode(os.fspath(path)))
        if not handle:
            raise OSError(f"qwen_win32_direct_open_utf8 failed for {path}")
        return int(handle)

    def close(self, handle: int) -> None:
        rc = int(self.lib.qwen_win32_direct_close(ctypes.c_void_p(handle)))
        if rc != 0:
            raise RuntimeError(f"direct close failed rc={rc}")

    def alignment(self, handle: int) -> dict[str, int]:
        h = ctypes.c_void_p(handle)
        return {
            "logical_sector": int(self.lib.qwen_win32_direct_logical_sector(h)),
            "physical_sector": int(self.lib.qwen_win32_direct_physical_sector(h)),
            "alignment": int(self.lib.qwen_win32_direct_alignment(h)),
        }

    def create_buffer(self, size: int, alignment: int) -> int:
        buf = self.lib.qwen_win32_buffer_create(size, alignment)
        if not buf:
            raise MemoryError(
                f"qwen_win32_buffer_create({size}, {alignment}) failed")
        return int(buf)

    def destroy_buffer(self, buffer: int) -> None:
        rc = int(self.lib.qwen_win32_buffer_destroy(ctypes.c_void_p(buffer)))
        if rc != 0:
            raise RuntimeError(f"buffer destroy failed rc={rc}")

    def buffer_ptr(self, buffer: int) -> int:
        ptr = self.lib.qwen_win32_buffer_ptr(ctypes.c_void_p(buffer))
        if not ptr:
            raise RuntimeError("native buffer pointer is null")
        return int(ptr)

    def read(
        self,
        handle: int,
        buffer: int,
        *,
        buffer_offset: int,
        file_offset: int,
        nbytes: int,
    ) -> int:
        return int(self.lib.qwen_win32_direct_read(
            ctypes.c_void_p(handle),
            ctypes.c_void_p(buffer),
            buffer_offset,
            file_offset,
            nbytes,
        ))

    def report(self, handle: int) -> dict[str, Any]:
        h = ctypes.c_void_p(handle)
        return {
            **self.alignment(handle),
            "no_buffering": bool(self.lib.qwen_win32_direct_no_buffering(h)),
            "overlapped": bool(self.lib.qwen_win32_direct_overlapped(h)),
            "calls": int(self.lib.qwen_win32_direct_calls(h)),
            "bytes": int(self.lib.qwen_win32_direct_bytes(h)),
            "seconds": float(self.lib.qwen_win32_direct_seconds(h)),
            "last_error": int(self.lib.qwen_win32_direct_last_error(h)),
        }


def _power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def synthetic_gate(library: Path) -> dict[str, Any]:
    api = Win32DirectIO(library)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "direct-io-fixture.bin"
        size = 8 * 1024 * 1024
        half = size // 2
        rng = random.Random(0x38D1)
        payload = rng.randbytes(size)
        source.write_bytes(payload)

        handle = api.open(source)
        buffer = 0
        try:
            align = api.alignment(handle)
            logical = align["logical_sector"]
            physical = align["physical_sector"]
            alignment = align["alignment"]
            if not (_power_of_two(logical)
                    and _power_of_two(physical)
                    and _power_of_two(alignment)):
                raise RuntimeError(f"non-power-of-two sector report: {align}")
            if alignment < logical or alignment < physical:
                raise RuntimeError(f"unsafe alignment report: {align}")
            if half % logical:
                raise RuntimeError(
                    f"fixture half {half} is not logical-sector aligned: {align}")

            buffer = api.create_buffer(size, alignment)
            ptr = api.buffer_ptr(buffer)
            if ptr % alignment:
                raise RuntimeError(
                    f"native buffer ptr {ptr:#x} is not {alignment}-aligned")

            rc0 = api.read(
                handle, buffer,
                buffer_offset=0, file_offset=0, nbytes=half)
            rc1 = api.read(
                handle, buffer,
                buffer_offset=half, file_offset=half, nbytes=half)
            if rc0 != 0 or rc1 != 0:
                report = api.report(handle)
                raise RuntimeError(
                    f"aligned direct reads failed rc0={rc0} rc1={rc1} "
                    f"report={report}")

            observed = ctypes.string_at(ptr, size)
            expected_sha = hashlib.sha256(payload).hexdigest()
            observed_sha = hashlib.sha256(observed).hexdigest()
            if observed_sha != expected_sha:
                raise RuntimeError(
                    f"direct read SHA mismatch {observed_sha} != {expected_sha}")

            calls_before_bad = api.report(handle)["calls"]
            bad = api.read(
                handle, buffer,
                buffer_offset=0, file_offset=1, nbytes=logical)
            if bad != -2:
                raise RuntimeError(
                    f"unaligned file offset should return -2, got {bad}")
            report = api.report(handle)
            if report["calls"] != calls_before_bad:
                raise RuntimeError("rejected unaligned read changed call counter")
            if not report["no_buffering"] or not report["overlapped"]:
                raise RuntimeError(f"Win32 mode flags missing: {report}")
            if report["calls"] != 2 or report["bytes"] != size:
                raise RuntimeError(f"unexpected direct counters: {report}")

            result = {
                "status": "PASS",
                "fixture_bytes": size,
                "sha256": observed_sha,
                "aligned_read_calls": report["calls"],
                "aligned_read_bytes": report["bytes"],
                "direct_read_seconds": report["seconds"],
                "logical_sector": logical,
                "physical_sector": physical,
                "alignment": alignment,
                "no_buffering": report["no_buffering"],
                "overlapped": report["overlapped"],
                "single_io_contract": True,
                "unaligned_rejection_rc": bad,
            }
            print(result)
            print("QWEN38_WIN32_DIRECT_IO_SYNTHETIC_PASS")
            print("QWEN38_WIN32_DIRECT_IO_NO_BUFFERING_ABI_PASS")
            return result
        finally:
            if buffer:
                api.destroy_buffer(buffer)
            api.close(handle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, required=True)
    args = ap.parse_args()
    synthetic_gate(args.lib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
