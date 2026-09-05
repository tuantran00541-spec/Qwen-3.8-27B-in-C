#!/usr/bin/env python3
"""Bitwise cross-platform probe for the exact 11-token Qwen3.8 text RoPE math."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys

ROPE_N_ROT = 64
ROPE_THETA = 10_000_000.0
PREFILL_POSITIONS = range(1, 11)  # pos=0 exits before any RoPE transcendental math.
RECORD = struct.Struct("<IIQQII")
FIELDS = ("pair", "position", "freq_f64", "angle_f64", "cos_f32", "sin_f32")


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _u32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _f32(x)))[0]


def _u64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def fingerprint(output: Path, summary: Path) -> dict:
    digest = hashlib.sha256()
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for pos in PREFILL_POSITIONS:
            for i in range(ROPE_N_ROT // 2):
                freq = ROPE_THETA ** (-(2.0 * i) / ROPE_N_ROT)
                angle = float(pos) * freq
                record = RECORD.pack(
                    i,
                    pos,
                    _u64(freq),
                    _u64(angle),
                    _u32(math.cos(angle)),
                    _u32(math.sin(angle)),
                )
                f.write(record)
                digest.update(record)
                count += 1
    payload = {
        "schema": "qwen38-cross-platform-rope-math-fingerprint-v1",
        "platform": sys.platform,
        "platform_detail": platform.platform(),
        "python": sys.version,
        "rope_theta": ROPE_THETA,
        "rope_n_rot": ROPE_N_ROT,
        "positions": list(PREFILL_POSITIONS),
        "records": count,
        "record_bytes": RECORD.size,
        "fingerprint_sha256": digest.hexdigest(),
        "output_bytes": output.stat().st_size,
        "arithmetic_change": False,
    }
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("QWEN38_CROSS_PLATFORM_ROPE_MATH_FINGERPRINT_PASS")
    return payload


def compare(linux: Path, windows: Path, output: Path) -> dict:
    left = linux.read_bytes()
    right = windows.read_bytes()
    if len(left) != len(right) or len(left) % RECORD.size:
        raise RuntimeError(
            f"RoPE fingerprint shape mismatch linux={len(left)} windows={len(right)} record={RECORD.size}")

    mismatch_counts = {name: 0 for name in FIELDS[2:]}
    first_mismatches: list[dict] = []
    records = len(left) // RECORD.size
    for idx in range(records):
        start = idx * RECORD.size
        lrec = RECORD.unpack_from(left, start)
        wrec = RECORD.unpack_from(right, start)
        if lrec[:2] != wrec[:2]:
            raise RuntimeError(f"RoPE corpus identity diverged at record {idx}")
        changed: dict[str, dict[str, str]] = {}
        for pos, name in enumerate(FIELDS[2:], start=2):
            if lrec[pos] != wrec[pos]:
                mismatch_counts[name] += 1
                width = 16 if pos <= 3 else 8
                changed[name] = {
                    "linux_bits": f"0x{lrec[pos]:0{width}x}",
                    "windows_bits": f"0x{wrec[pos]:0{width}x}",
                }
        if changed and len(first_mismatches) < 24:
            first_mismatches.append({
                "record": idx,
                "pair": lrec[0],
                "position": lrec[1],
                "changed": changed,
            })

    total = sum(mismatch_counts.values())
    payload = {
        "schema": "qwen38-cross-platform-rope-math-compare-v1",
        "records": records,
        "bitwise_match": total == 0,
        "mismatch_counts": mismatch_counts,
        "first_mismatches": first_mismatches,
        "claim": "diagnostic of the exact 11-token text RoPE pow/cos/sin path only",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if total:
        print("QWEN38_CROSS_PLATFORM_ROPE_MATH_DIVERGENCE_FOUND")
        raise SystemExit("Windows RoPE transcendental path is not bitwise identical to Linux")
    print("QWEN38_CROSS_PLATFORM_ROPE_MATH_BITWISE_PASS")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fp = sub.add_parser("fingerprint")
    p_fp.add_argument("--output", type=Path, required=True)
    p_fp.add_argument("--summary", type=Path, required=True)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--linux", type=Path, required=True)
    p_cmp.add_argument("--windows", type=Path, required=True)
    p_cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "fingerprint":
        fingerprint(args.output, args.summary)
    else:
        compare(args.linux, args.windows, args.output)


if __name__ == "__main__":
    main()
