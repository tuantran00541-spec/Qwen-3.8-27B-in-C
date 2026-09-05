#!/usr/bin/env python3
"""Synthetic CPU-only groupwise weight quantization gate for Qwen3.8 research.

This is deliberately below real-model inference. It validates the packed format,
odd-bit packing (Q5), group scales, dequantization, and linear error metrics before
we spend CI disk/time quantizing official 27B weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def quantize_groupwise_symmetric(weight: torch.Tensor, bits: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if bits not in (4, 5):
        raise ValueError("bits must be 4 or 5")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    w = weight.float().contiguous()
    shape = tuple(w.shape)
    flat = w.reshape(-1)
    groups = math.ceil(flat.numel() / group_size)
    padded = groups * group_size
    if padded != flat.numel():
        flat = F.pad(flat, (0, padded - flat.numel()))
    grouped = flat.reshape(groups, group_size)
    qmax = (1 << (bits - 1)) - 1
    scale = grouped.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / qmax
    q = torch.round(grouped / scale[:, None]).clamp(-qmax, qmax).to(torch.int16)
    unsigned = (q + qmax).to(torch.uint8).reshape(-1)
    return unsigned[:padded], scale, shape


def pack_codes(codes: torch.Tensor, bits: int) -> bytes:
    mask = (1 << bits) - 1
    out = bytearray(math.ceil(codes.numel() * bits / 8))
    acc = 0
    acc_bits = 0
    pos = 0
    for value in codes.tolist():
        acc |= (int(value) & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out[pos] = acc & 0xFF
            pos += 1
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out[pos] = acc & 0xFF
    return bytes(out)


def unpack_codes(blob: bytes, count: int, bits: int) -> torch.Tensor:
    mask = (1 << bits) - 1
    out = torch.empty(count, dtype=torch.uint8)
    acc = 0
    acc_bits = 0
    src = 0
    for i in range(count):
        while acc_bits < bits:
            acc |= blob[src] << acc_bits
            src += 1
            acc_bits += 8
        out[i] = acc & mask
        acc >>= bits
        acc_bits -= bits
    return out


def dequantize(codes: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...], bits: int, group_size: int) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    signed = codes.to(torch.int16) - qmax
    grouped = signed.reshape(-1, group_size).float() * scale[:, None]
    return grouped.reshape(-1)[: math.prod(shape)].reshape(shape)


def metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict:
    diff = cand.float() - ref.float()
    denom = torch.linalg.vector_norm(ref.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": float(diff.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff))),
        "relative_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(F.cosine_similarity(ref.float().flatten(), cand.float().flatten(), dim=0)),
    }


def run(bits: int, group_size: int, rows: int, cols: int, seed: int) -> dict:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    weight = torch.randn((rows, cols), generator=gen, dtype=torch.float32) * 0.02
    x = torch.randn((3, cols), generator=gen, dtype=torch.float32)
    codes, scales, shape = quantize_groupwise_symmetric(weight, bits, group_size)
    packed = pack_codes(codes, bits)
    unpacked = unpack_codes(packed, codes.numel(), bits)
    if not torch.equal(codes, unpacked):
        raise RuntimeError(f"Q{bits} pack roundtrip mismatch")
    restored = dequantize(unpacked, scales, shape, bits, group_size)
    ref_y = F.linear(x, weight)
    q_y = F.linear(x, restored)
    scale_bytes = scales.numel() * 2  # production target stores scales as FP16/BF16
    logical_bytes = len(packed) + scale_bytes
    bf16_bytes = weight.numel() * 2
    return {
        "bits": bits,
        "group_size": group_size,
        "shape": list(shape),
        "packed_code_bytes": len(packed),
        "scale_bytes_target": scale_bytes,
        "logical_quant_bytes": logical_bytes,
        "bf16_bytes": bf16_bytes,
        "compression_vs_bf16": bf16_bytes / logical_bytes,
        "weight_error": metrics(weight, restored),
        "linear_error": metrics(ref_y, q_y),
        "pack_roundtrip_exact": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--rows", type=int, default=257)
    p.add_argument("--cols", type=int, default=513)
    p.add_argument("--seed", type=int, default=20260831)
    a = p.parse_args()
    results = [run(bits, a.group_size, a.rows, a.cols, a.seed) for bits in (5, 4)]
    ok = all(r["pack_roundtrip_exact"] and math.isfinite(r["linear_error"]["relative_l2"]) for r in results)
    result = {
        "schema": "qwen38-k3-quant-sanity-v1",
        "status": "PASS" if ok else "FAIL",
        "method": "symmetric per-group weight-only prototype",
        "scale_storage_target": "FP16/BF16",
        "results": results,
        "real_weights_attempted": False,
        "generation_attempted": False,
        "quality_claimed": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
