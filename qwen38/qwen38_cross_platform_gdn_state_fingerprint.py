#!/usr/bin/env python3
"""Cross-platform bitwise fingerprint for the exact recurrent GDN state kernel.

Diagnostic only: this drives the current ``gdn_state_ar.c`` arithmetic through
its exported single-row and sequential-batch ABIs using fixed raw F32 fixtures.
The gate corpus deliberately includes inputs already proven to produce different
``expf`` bits between glibc and Windows UCRT.

The fixture isolates the decay boundary: K, V and beta are zero, so the state
update cannot numerically hide a one-ULP ``expf`` difference behind the later
outer-product update. Q remains nonzero so the returned row also observes the
decayed state. The output captures both mutated persistent state and returned
rows so a Windows compatibility build can be compared directly with Linux
before touching the full GGUF gate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct

HEADS = 48
DIM = 128
VALUE_DIM = HEADS * DIM
STATE_VALUES = HEADS * DIM * DIM
ROWS = 2

# Exact negative F32 inputs where the existing scalar fingerprint observed an
# expf bit mismatch between Linux/glibc and Windows UCRT.
GATE_BITS = (
    0xC1C9F000,  # -25.2421875
    0xC1C2B800,  # -24.33984375
    0xC1914000,  # -18.15625
    0xC15E1000,  # -13.87890625
    0xC148F000,  # -12.55859375
    0xC1433000,  # -12.19921875
    0xC19ADEAC,  # -19.358726501464844
    0xC1C8C9D4,  # -25.098548889160156
)

# All remaining fixtures are exact binary fractions encoded as raw F32 bits,
# avoiding Python/libm differences during fixture construction.
STATE_BITS = (
    0x3F800000,  # 1
    0xBF000000,  # -1/2
    0x3E800000,  # 1/4
    0xBE000000,  # -1/8
    0x3D800000,  # 1/16
    0x00000000,
)
Q_BITS = (0x3C000000, 0xBC000000, 0x3B800000, 0xBB800000)  # +/-1/128, +/-1/256
ZERO_BITS = (0x00000000,)


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _fill_raw_f32(count: int, bits: tuple[int, ...], salt: int = 0):
    arr = (ctypes.c_float * count)()
    for i in range(count):
        arr[i] = _f32_from_bits(bits[(i + salt) % len(bits)])
    return arr


def _bytes(obj) -> bytes:
    return ctypes.string_at(ctypes.addressof(obj), ctypes.sizeof(obj))


def _bind(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path.resolve()))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.qwen_gdn_ar_step_f32.argtypes = [fp, fp, fp, fp, fp, fp, fp]
    lib.qwen_gdn_ar_step_f32.restype = ctypes.c_int
    lib.qwen_gdn_ar_batch_f32.argtypes = [
        fp, ctypes.c_size_t, fp, fp, fp, fp, fp, fp,
    ]
    lib.qwen_gdn_ar_batch_f32.restype = ctypes.c_int
    return lib


def fingerprint(lib_path: Path, output: Path, summary: Path) -> dict:
    lib = _bind(lib_path)

    initial = _fill_raw_f32(STATE_VALUES, STATE_BITS)
    q = _fill_raw_f32(ROWS * VALUE_DIM, Q_BITS, 1)
    # Decay-only fixture: no outer-product update may overwrite the direct
    # state *= expf(gate) observation.  Nonzero Q still propagates state bits
    # into the returned output row.
    k = _fill_raw_f32(ROWS * VALUE_DIM, ZERO_BITS)
    v = _fill_raw_f32(ROWS * VALUE_DIM, ZERO_BITS)
    gate = _fill_raw_f32(ROWS * HEADS, GATE_BITS)
    beta = _fill_raw_f32(ROWS * HEADS, ZERO_BITS)

    StepState = ctypes.c_float * STATE_VALUES
    StepOut = ctypes.c_float * VALUE_DIM
    BatchOut = ctypes.c_float * (ROWS * VALUE_DIM)

    step_state = StepState.from_buffer_copy(_bytes(initial))
    step_out = StepOut()
    rc = lib.qwen_gdn_ar_step_f32(
        step_state,
        ctypes.cast(q, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(k, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(v, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(gate, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(beta, ctypes.POINTER(ctypes.c_float)),
        step_out,
    )
    if rc != 0:
        raise RuntimeError(f"qwen_gdn_ar_step_f32 rc={rc}")

    batch_state = StepState.from_buffer_copy(_bytes(initial))
    batch_out = BatchOut()
    rc = lib.qwen_gdn_ar_batch_f32(
        batch_state,
        ROWS,
        q,
        k,
        v,
        gate,
        beta,
        batch_out,
    )
    if rc != 0:
        raise RuntimeError(f"qwen_gdn_ar_batch_f32 rc={rc}")

    chunks = (
        ("step_state", _bytes(step_state)),
        ("step_out", _bytes(step_out)),
        ("batch_state", _bytes(batch_state)),
        ("batch_out", _bytes(batch_out)),
    )
    payload_bytes = b"".join(data for _, data in chunks)
    output.write_bytes(payload_bytes)

    offsets: dict[str, dict[str, int | str]] = {}
    pos = 0
    for name, data in chunks:
        offsets[name] = {
            "offset": pos,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        pos += len(data)

    payload = {
        "schema": "qwen38-cross-platform-gdn-state-fingerprint-v2",
        "library": str(lib_path),
        "rows": ROWS,
        "heads": HEADS,
        "dim": DIM,
        "fixture": "decay-only-k-v-beta-zero-q-nonzero",
        "gate_bits": [f"0x{x:08x}" for x in GATE_BITS],
        "segments": offsets,
        "fingerprint_bytes": len(payload_bytes),
        "fingerprint_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "arithmetic_change": False,
    }
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("QWEN38_GDN_STATE_CROSS_PLATFORM_FINGERPRINT_PASS")
    return payload


def compare(linux: Path, windows: Path, *, require_match: bool) -> bool:
    left = linux.read_bytes()
    right = windows.read_bytes()
    if len(left) != len(right):
        raise RuntimeError(f"shape mismatch linux={len(left)} windows={len(right)}")
    match = left == right
    first = None
    if not match:
        first = next(i for i, (a, b) in enumerate(zip(left, right)) if a != b)
    payload = {
        "schema": "qwen38-cross-platform-gdn-state-compare-v2",
        "bytes": len(left),
        "bitwise_match": match,
        "first_mismatch_byte": first,
        "linux_sha256": hashlib.sha256(left).hexdigest(),
        "windows_sha256": hashlib.sha256(right).hexdigest(),
        "required_match": require_match,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if require_match:
        if not match:
            raise SystemExit("required GDN state fingerprint match failed")
        print("QWEN38_GDN_STATE_CROSS_PLATFORM_BITWISE_MATCH")
    else:
        if match:
            raise SystemExit("expected baseline GDN state divergence was not observed")
        print("QWEN38_GDN_STATE_CROSS_PLATFORM_DIVERGENCE_FOUND")
    return match


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fp = sub.add_parser("fingerprint")
    p_fp.add_argument("--lib", type=Path, required=True)
    p_fp.add_argument("--output", type=Path, required=True)
    p_fp.add_argument("--summary", type=Path, required=True)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--linux", type=Path, required=True)
    p_cmp.add_argument("--windows", type=Path, required=True)
    p_cmp.add_argument("--require-match", action="store_true")

    args = parser.parse_args()
    if args.cmd == "fingerprint":
        fingerprint(args.lib, args.output, args.summary)
    else:
        compare(args.linux, args.windows, require_match=args.require_match)


if __name__ == "__main__":
    main()
