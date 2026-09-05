#!/usr/bin/env python3
"""Real-GGUF Q4_0 exactness gate on Qwen3.8's MTP entry projection.

This does not implement MTP semantics yet.  It proves that the isolated Q4_0
bridge consumes the pinned GGUF bytes for blk.64.nextn.eh_proj.weight and that
its AVX2 result is bitwise identical to the scalar GGML-order reference for an
entire 10240 -> 5120 matrix-vector product.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import resource
import time

from gguf_stream import parse_gguf
from qwen35_gdn_quant_layer_gate import SHA256

TENSOR = "blk.64.nextn.eh_proj.weight"
EXPECTED_SHAPE = [10240, 5120]
EXPECTED_TYPE = "Q4_0"


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bind_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.qwen_quantize_q8_0_scalar.argtypes = [fp, ctypes.c_size_t, u8p, ctypes.c_size_t]
    lib.qwen_quantize_q8_0_scalar.restype = ctypes.c_int
    for name in ("qwen_matvec_q4_0_q8_0_reference", "qwen_matvec_q4_0_q8_0_scalar"):
        fn = getattr(lib, name)
        fn.argtypes = [u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                       u8p, ctypes.c_size_t, fp]
        fn.restype = ctypes.c_int
    return lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--native-lib", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    digest = sha256_file(args.model)
    if digest != SHA256:
        raise SystemExit(f"model sha mismatch: {digest}")

    directory = parse_gguf(args.model)
    tensors = directory.by_name()
    if TENSOR not in tensors:
        raise SystemExit(f"missing {TENSOR}")
    tensor = tensors[TENSOR]
    if tensor.type_name != EXPECTED_TYPE or list(tensor.shape) != EXPECTED_SHAPE:
        raise SystemExit(f"bad contract: type={tensor.type_name} shape={list(tensor.shape)}")

    ne0, rows = map(int, tensor.shape)
    with args.model.open("rb", buffering=0) as f:
        f.seek(int(tensor.data_offset))
        raw = f.read(int(tensor.nbytes))
    if len(raw) != int(tensor.nbytes):
        raise SystemExit("short tensor read")
    weights = bytearray(raw)

    # Deterministic finite activation covering positive/negative ranges.  The
    # goal is kernel equivalence on real matrix bytes, not a model-semantic
    # checkpoint; that comes in the following MTP gate.
    x = (ctypes.c_float * ne0)(
        *(((i * 37) % 1021 - 510) / 173.0 for i in range(ne0))
    )
    activation_bytes = (ne0 // 32) * 34
    activation = (ctypes.c_uint8 * activation_bytes)()
    lib = bind_lib(args.native_lib)
    rc = lib.qwen_quantize_q8_0_scalar(x, ne0, activation, activation_bytes)
    if rc != 0:
        raise SystemExit(f"q8 activation quantization rc={rc}")

    w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
    ref = (ctypes.c_float * rows)()
    avx = (ctypes.c_float * rows)()

    t0 = time.monotonic()
    rc = lib.qwen_matvec_q4_0_q8_0_reference(
        w_arr, len(weights), rows, ne0, activation, activation_bytes, ref)
    ref_s = time.monotonic() - t0
    if rc != 0:
        raise SystemExit(f"reference matvec rc={rc}")

    t0 = time.monotonic()
    rc = lib.qwen_matvec_q4_0_q8_0_scalar(
        w_arr, len(weights), rows, ne0, activation, activation_bytes, avx)
    avx_s = time.monotonic() - t0
    if rc != 0:
        raise SystemExit(f"avx2 matvec rc={rc}")

    ref_bytes = ctypes.string_at(ctypes.addressof(ref), ctypes.sizeof(ref))
    avx_bytes = ctypes.string_at(ctypes.addressof(avx), ctypes.sizeof(avx))
    exact = ref_bytes == avx_bytes
    payload = {
        "schema": "qwen38-mtp-q4-real-v1",
        "status": "PASS" if exact else "FAIL",
        "model_sha256": digest,
        "tensor": TENSOR,
        "tensor_type": tensor.type_name,
        "tensor_shape": list(tensor.shape),
        "tensor_bytes": int(tensor.nbytes),
        "activation_bytes": activation_bytes,
        "output_rows": rows,
        "output_sha256": hashlib.sha256(avx_bytes).hexdigest(),
        "bitwise_exact": exact,
        "reference_seconds": ref_s,
        "avx2_seconds": avx_s,
        "speedup_reference_over_avx2": ref_s / avx_s if avx_s else None,
        "max_rss_gib": rss_gib(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not exact:
        raise SystemExit("real Q4 matvec is not bitwise exact")
    print("QWEN38_MTP_Q4_REAL_EH_PROJ_BITWISE_PASS")


if __name__ == "__main__":
    main()
