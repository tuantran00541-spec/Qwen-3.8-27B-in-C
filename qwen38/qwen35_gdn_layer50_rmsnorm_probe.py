#!/usr/bin/env python3
"""Probe pinned ggml RMSNorm arithmetic against the isolated layer-50 cliff.

This intentionally monkeypatches only the Python RMSNorm helper used by the
layer-50 microscope.  It does not change the production/full64 executor.  The
pinned ggml CPU implementation accumulates double precision sums of F32-rounded
squares, then returns to F32 for mean, sqrtf/scale and output multiplication.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_gdn_layer50_microscope as microscope


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def ggml_rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = gdn.RMS_EPS) -> list[float]:
    if len(x) != len(weight):
        raise ValueError("RMSNorm shape mismatch")
    # Pinned ops.cpp: sum += (ggml_float)(x[i] * x[i]); ggml_float is double,
    # but x*x is an F32 operation before the cast to the accumulator.
    sum_sq = 0.0
    for value in x:
        vf = f32(value)
        sum_sq += float(f32(vf * vf))
    mean = f32(sum_sq / len(x))
    mean_eps = f32(mean + f32(eps))
    sqrt_mean = f32(math.sqrt(mean_eps))
    scale = f32(1.0 / sqrt_mean)

    out: list[float] = []
    for value, w in zip(x, weight):
        scaled = f32(f32(value) * scale)
        out.append(f32(scaled * f32(w)))
    return out


def sanity() -> None:
    # The probe must actually exercise F32 product rounding rather than the old
    # all-double Python expression.
    x = f32(1.0000001192092896)
    product = f32(x * x)
    if product != 1.000000238418579:
        raise SystemExit(f"unexpected F32 product probe {product!r}")
    y = ggml_rms_norm([x, -x], [1.0, 1.0], 1e-6)
    if len(y) != 2 or not all(math.isfinite(v) for v in y):
        raise SystemExit("ggml RMSNorm probe sanity failed")
    print(json.dumps({
        "schema": "qwen38-ggml-rmsnorm-arithmetic-sanity-v1",
        "status": "PASS",
        "f32_square": product,
        "ggml_float_accumulator": "double",
        "square_before_accumulate": "F32",
        "mean_scale_output": "F32",
    }, indent=2))


def execute(model: Path, native_lib: Path, oracle: Path, work_dir: Path, output: Path) -> dict:
    original = gdn.rms_norm
    gdn.rms_norm = ggml_rms_norm
    try:
        result = microscope.execute(model, native_lib, oracle, work_dir, output)
    finally:
        gdn.rms_norm = original
    result["schema"] = "qwen38-gdn-layer50-ggml-rmsnorm-probe-v1"
    result["rmsnorm_arithmetic"] = {
        "source": "pinned llama.cpp ggml CPU ops.cpp",
        "ggml_float_accumulator": "double",
        "input_square": "F32 then cast to double",
        "mean": "F32",
        "sqrt_scale": "sqrtf/F32",
        "output_mul": "F32",
        "production_runtime_modified": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--oracle", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "sanity":
        sanity()
    else:
        a.work_dir.mkdir(parents=True, exist_ok=True)
        execute(a.model, a.native_lib, a.oracle, a.work_dir, a.output)


if __name__ == "__main__":
    main()
