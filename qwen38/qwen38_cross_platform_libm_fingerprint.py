#!/usr/bin/env python3
"""Cross-platform fingerprint for exact libm-dependent Qwen3.8 scalar helpers.

This is diagnostic-only.  It does not change decoder arithmetic.  The same
fixed F32 input corpus is evaluated on Linux/glibc and Windows/UCRT (or an
explicit diagnostic expf shim) so hosted CI can characterize bitwise math
portability before another full GGUF run is spent on a cross-platform anchor.

Besides the expf-derived gate helpers, this probe fingerprints the remaining
sqrt-sensitive boundaries used by the current-best path:
  * sqrtf(mean_eps), as used by the native GDN output RMSNorm gate;
  * sqrt((double)mean_eps) rounded to F32, as used by native RMSNorm;
  * Python double sqrt followed by the proven F32 inverse-sqrt boundary;
  * an L2-like double sqrt/divide followed by the F32 marshal boundary used
    before native GDN repeat/scale.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
from pathlib import Path
import platform
from types import SimpleNamespace
import struct
import sys

if sys.platform == "win32":
    from qwen38_win32_bootstrap import install_resource_compat

    install_resource_compat()

import qwen35_k3_full64_ggml_exact as exact

FIELDS = (
    "input",
    "expf",
    "sigmoid",
    "softplus",
    "silu",
    "sqrtf",
    "sqrt64_f32",
    "inv_sqrt_f32",
    "l2_probe_f32",
)
RECORD = struct.Struct("<" + "I" * len(FIELDS))
RANDOM_CASES = 262_144
WIDE_RANDOM_CASES = 262_144
GRID_DENOM = 256
GRID_LIMIT = 8_192


def _bind_platform_expf(expf_lib: Path | None = None, expf_symbol: str | None = None) -> str:
    if expf_lib is not None:
        symbol = expf_symbol or "qwen38_glibc_expf_compat"
        lib = ctypes.CDLL(str(expf_lib.resolve()))
        fn = getattr(lib, symbol)
        fn.argtypes = [ctypes.c_float]
        fn.restype = ctypes.c_float
        # Keep the CDLL alive through the namespace while exposing exactly the
        # .expf attribute expected by the proven exact wrapper.
        exact._LIBM = SimpleNamespace(expf=fn, _library=lib)
        return f"{expf_lib.resolve()}!{symbol}"
    if expf_symbol is not None:
        raise ValueError("--expf-symbol requires --expf-lib")
    if sys.platform == "win32":
        ucrt = ctypes.CDLL("ucrtbase.dll")
        ucrt.expf.argtypes = [ctypes.c_float]
        ucrt.expf.restype = ctypes.c_float
        exact._LIBM = ucrt
        return "ucrtbase.dll!expf"
    return str(getattr(exact, "_name", None) or "python-math-exp-fallback")


def _bind_platform_sqrt():
    if sys.platform == "win32":
        name = "ucrtbase.dll"
    else:
        name = ctypes.util.find_library("m")
        if not name:
            raise RuntimeError("platform libm was not found for sqrt fingerprint")
    lib = ctypes.CDLL(name)
    lib.sqrt.argtypes = [ctypes.c_double]
    lib.sqrt.restype = ctypes.c_double
    lib.sqrtf.argtypes = [ctypes.c_float]
    lib.sqrtf.restype = ctypes.c_float
    return lib, f"{name}!sqrt/sqrtf"


def _f32(x: float) -> float:
    return exact.f32(float(x))


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _f32(x)))[0]


def _from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def _softplusf(x: float) -> float:
    xf = _f32(x)
    if xf > 20.0:
        return xf
    if xf < -20.0:
        return _f32(exact.expf(xf))
    return _f32(math.log1p(float(exact.expf(xf))))


def _siluf(x: float) -> float:
    return exact.mul_f32(x, exact.sigmoid_f32(x))


def _sqrt_fields(x: float, sqrt_lib) -> tuple[int, int, int, int]:
    xf = _f32(x)

    # Native RMSNorm/GDN gate both see non-negative norm statistics.  Keep the
    # probe in a realistic finite range and round the epsilon addition to F32
    # before either sqrt style, matching the current exact RMSNorm boundaries.
    mean = _f32(abs(xf))
    mean_eps = _f32(mean + _f32(1e-6))

    sqrtf_value = float(sqrt_lib.sqrtf(ctypes.c_float(mean_eps)))
    sqrt64_value = float(sqrt_lib.sqrt(ctypes.c_double(float(mean_eps))))
    sqrt64_f32 = _f32(sqrt64_value)
    inv_sqrt_f32 = _f32(1.0 / _f32(math.sqrt(float(mean_eps))))

    # The recurrent q/k L2 normalization is still Python double arithmetic:
    # denom = sqrt(fsum(v*v)); each normalized double is later marshaled to F32
    # by array('f') in ExactGDNRepeatScale.  This scalar surrogate deliberately
    # retains the double sqrt/divide boundary and fingerprints the resulting F32.
    norm_sq = abs(float(xf)) * 128.0 + 1.0e-12
    numerator = _f32(xf * 0.03125 + 0.75)
    l2_probe = _f32(float(numerator) / math.sqrt(norm_sq))

    return (
        _bits(sqrtf_value),
        _bits(sqrt64_f32),
        _bits(inv_sqrt_f32),
        _bits(l2_probe),
    )


def _corpus_bits() -> list[int]:
    fixed = (
        -104.0, -103.98, -103.97, -103.3, -100.0, -90.0, -80.0,
        -40.0, -32.0, -20.0, -10.0, -5.0, -2.0, -1.0, -0.5,
        -0.125, -0.0, 0.0, 0.125, 0.5, 1.0, 2.0, 5.0, 10.0,
        20.0, 32.0, 40.0, 80.0, 88.0, 88.7, 88.72, 88.73, 89.0,
    )
    values: list[int] = [_bits(x) for x in fixed]
    values.extend(_bits(i / GRID_DENOM) for i in range(-GRID_LIMIT, GRID_LIMIT + 1))

    # Dense random mantissa sample around the values most likely in gates.
    state = 0x6D2B79F5
    scale = 64.0 / float(1 << 24)
    for _ in range(RANDOM_CASES):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values.append(_bits(((state >> 8) * scale) - 32.0))

    # Separate wide sample spans expf's useful finite F32 domain, including
    # subnormal underflow and overflow boundaries relevant to saturated gates.
    state = 0xA341316C
    wide_scale = 193.0 / float(1 << 24)  # [-104, 89)
    for _ in range(WIDE_RANDOM_CASES):
        state = (22695477 * state + 1) & 0xFFFFFFFF
        values.append(_bits(((state >> 8) * wide_scale) - 104.0))

    seen: set[int] = set()
    unique: list[int] = []
    for bits in values:
        if bits not in seen:
            seen.add(bits)
            unique.append(bits)
    return unique


def fingerprint(
    output: Path,
    summary: Path,
    *,
    expf_lib: Path | None = None,
    expf_symbol: str | None = None,
) -> dict:
    backend = _bind_platform_expf(expf_lib, expf_symbol)
    sqrt_lib, sqrt_backend = _bind_platform_sqrt()
    corpus = _corpus_bits()
    digest = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for input_bits in corpus:
            x = _from_bits(input_bits)
            sqrtf_bits, sqrt64_bits, inv_sqrt_bits, l2_probe_bits = _sqrt_fields(x, sqrt_lib)
            record = RECORD.pack(
                input_bits,
                _bits(exact.expf(x)),
                _bits(exact.sigmoid_f32(x)),
                _bits(_softplusf(x)),
                _bits(_siluf(x)),
                sqrtf_bits,
                sqrt64_bits,
                inv_sqrt_bits,
                l2_probe_bits,
            )
            f.write(record)
            digest.update(record)
    payload = {
        "schema": "qwen38-cross-platform-libm-fingerprint-v2",
        "platform": sys.platform,
        "platform_detail": platform.platform(),
        "python": sys.version,
        "expf_backend": backend,
        "sqrt_backend": sqrt_backend,
        "fields": list(FIELDS),
        "cases": len(corpus),
        "record_bytes": RECORD.size,
        "fingerprint_sha256": digest.hexdigest(),
        "output_bytes": output.stat().st_size,
        "arithmetic_change": False,
    }
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("QWEN38_CROSS_PLATFORM_LIBM_FINGERPRINT_PASS")
    return payload


def compare(
    linux: Path,
    windows: Path,
    output: Path,
    *,
    required_fields: tuple[str, ...] = (),
) -> dict:
    unknown = set(required_fields) - set(FIELDS[1:])
    if unknown:
        raise ValueError(f"unknown required fields: {sorted(unknown)}")
    left = linux.read_bytes()
    right = windows.read_bytes()
    if len(left) != len(right) or len(left) % RECORD.size:
        raise RuntimeError(
            f"fingerprint shape mismatch linux={len(left)} windows={len(right)} record={RECORD.size}")

    mismatch_counts = {name: 0 for name in FIELDS[1:]}
    first_mismatches: list[dict] = []
    cases = len(left) // RECORD.size
    for idx in range(cases):
        start = idx * RECORD.size
        lrec = RECORD.unpack_from(left, start)
        wrec = RECORD.unpack_from(right, start)
        if lrec[0] != wrec[0]:
            raise RuntimeError(f"input corpus diverged at case {idx}: {lrec[0]:08x} != {wrec[0]:08x}")
        changed: dict[str, dict[str, str]] = {}
        for pos, name in enumerate(FIELDS[1:], start=1):
            if lrec[pos] != wrec[pos]:
                mismatch_counts[name] += 1
                changed[name] = {
                    "linux_bits": f"0x{lrec[pos]:08x}",
                    "windows_bits": f"0x{wrec[pos]:08x}",
                }
        if changed and len(first_mismatches) < 24:
            first_mismatches.append({
                "case": idx,
                "input_bits": f"0x{lrec[0]:08x}",
                "input_f32": _from_bits(lrec[0]),
                "changed": changed,
            })

    total = sum(mismatch_counts.values())
    required_match = all(mismatch_counts[name] == 0 for name in required_fields)
    payload = {
        "schema": "qwen38-cross-platform-libm-compare-v2",
        "cases": cases,
        "bitwise_match": total == 0,
        "mismatch_counts": mismatch_counts,
        "required_fields": list(required_fields),
        "required_fields_match": required_match,
        "first_mismatches": first_mismatches,
        "claim": (
            "diagnostic comparison only; a divergence identifies a platform-libm candidate "
            "but does not by itself prove the full-model hidden-anchor root cause"
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if total:
        print("QWEN38_CROSS_PLATFORM_LIBM_DIVERGENCE_FOUND")
    else:
        print("QWEN38_CROSS_PLATFORM_LIBM_BITWISE_MATCH")
    if required_fields:
        if not required_match:
            raise SystemExit(
                "required fields differ: "
                + ", ".join(f"{name}={mismatch_counts[name]}" for name in required_fields)
            )
        print("QWEN38_CROSS_PLATFORM_LIBM_REQUIRED_FIELDS_MATCH " + ",".join(required_fields))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fp = sub.add_parser("fingerprint")
    p_fp.add_argument("--output", type=Path, required=True)
    p_fp.add_argument("--summary", type=Path, required=True)
    p_fp.add_argument("--expf-lib", type=Path)
    p_fp.add_argument("--expf-symbol")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--linux", type=Path, required=True)
    p_cmp.add_argument("--windows", type=Path, required=True)
    p_cmp.add_argument("--output", type=Path, required=True)
    p_cmp.add_argument(
        "--require-fields",
        default="",
        help="comma-separated result fields that must be bitwise identical",
    )
    args = parser.parse_args()
    if args.cmd == "fingerprint":
        fingerprint(
            args.output,
            args.summary,
            expf_lib=args.expf_lib,
            expf_symbol=args.expf_symbol,
        )
    else:
        required = tuple(x.strip() for x in args.require_fields.split(",") if x.strip())
        compare(args.linux, args.windows, args.output, required_fields=required)


if __name__ == "__main__":
    main()
