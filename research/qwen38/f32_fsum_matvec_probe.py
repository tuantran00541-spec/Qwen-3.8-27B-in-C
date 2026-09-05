#!/usr/bin/env python3
"""Bitwise oracle for the native CPython-style F32 fsum matvec probe."""
from __future__ import annotations

import argparse
import ctypes
import math
import random
import struct
import time


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def dbytes(x: float) -> bytes:
    return struct.pack("<d", float(x))


def load(path: str):
    lib = ctypes.CDLL(path)
    lib.qwen_matvec_f32_fsum_exact.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.qwen_matvec_f32_fsum_exact.restype = ctypes.c_int
    return lib


def make_case(seed: int, rows: int, n: int, wide: bool):
    rng = random.Random(seed)
    weights: list[float] = []
    for _ in range(rows * n):
        v = rng.uniform(-3.0, 3.0)
        weights.append(f32(v))
    x: list[float] = []
    for i in range(n):
        if wide:
            mant = rng.uniform(-1.0, 1.0)
            exp = rng.randint(-28, 28)
            v = math.ldexp(mant, exp)
        else:
            v = rng.uniform(-6.0, 6.0)
        # Deliberately keep activation as binary64: the production Python
        # f32_matvec receives Python floats, not forced-F32 activations.
        if i % 257 == 0:
            v = -v
        x.append(v)
    return weights, x


def run_case(lib, seed: int, rows: int, n: int, wide: bool) -> tuple[float, float]:
    weights, x = make_case(seed, rows, n, wide)
    w_arr = (ctypes.c_float * len(weights))(*weights)
    x_arr = (ctypes.c_double * n)(*x)
    out = (ctypes.c_double * rows)()

    t0 = time.perf_counter()
    ref = [
        math.fsum(float(weights[r*n + i]) * float(x[i]) for i in range(n))
        for r in range(rows)
    ]
    py_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rc = lib.qwen_matvec_f32_fsum_exact(w_arr, rows, n, x_arr, out)
    c_s = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"native fsum matvec rc={rc}")

    for r, expected in enumerate(ref):
        got = float(out[r])
        if dbytes(expected) != dbytes(got):
            raise RuntimeError(
                f"double-bitwise mismatch row={r} expected={expected.hex()} got={got.hex()}"
            )
    return py_s, c_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    args = ap.parse_args()
    lib = load(args.lib)

    timings = []
    cases = [
        (0x1355, 48, 5120, False),
        (0x3530, 48, 5120, True),
        (0x38, 17, 256, False),
        (0x27, 17, 256, True),
    ]
    for case in cases:
        timings.append(run_case(lib, *case))

    # Explicit cancellation-heavy rows; this exercises the partials collapse
    # and half-even correction rather than only benign random sums.
    n = 5120
    weights = [f32(1.0 if i % 2 == 0 else -1.0) for i in range(n)]
    x = [1.0e16 if i % 4 < 2 else 1.0 for i in range(n)]
    w_arr = (ctypes.c_float * n)(*weights)
    x_arr = (ctypes.c_double * n)(*x)
    out = (ctypes.c_double * 1)()
    ref = math.fsum(float(weights[i]) * float(x[i]) for i in range(n))
    rc = lib.qwen_matvec_f32_fsum_exact(w_arr, 1, n, x_arr, out)
    if rc != 0 or dbytes(ref) != dbytes(float(out[0])):
        raise RuntimeError("cancellation case mismatch")

    py_s, c_s = timings[0]
    print(f"model-shape Python math.fsum={py_s:.6f}s native={c_s:.6f}s speedup={py_s/c_s:.4f}x")
    print("QWEN38_F32_NATIVE_FSUM_DOUBLE_BITWISE_PASS")


if __name__ == "__main__":
    main()
