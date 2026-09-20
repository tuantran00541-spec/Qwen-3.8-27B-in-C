#!/usr/bin/env python3
from __future__ import annotations

from array import array
import math
from pathlib import Path
import struct
import sys
import time

from bonsai2_quant_runtime import Bonsai2NativeRuntime


HIDDEN = 512
QKV_ROWS = 257
GATE_ROWS = 193
SMALL_ROWS = 8

REAL_HIDDEN = 5120
REAL_QKV_ROWS = 10240
REAL_GATE_ROWS = 6144
REAL_SMALL_ROWS = 48


def ptq1_weights(rows: int, n: int, salt: int) -> memoryview:
    row_bytes = (n // 128) * 28
    raw = bytearray(rows * row_bytes)
    for r in range(rows):
        for b in range(n // 128):
            off = r * row_bytes + b * 28
            for j in range(24):
                raw[off + j] = (r * 17 + b * 29 + j * 11 + salt) % 243
            raw[off + 24] = (r * 7 + b * 13 + salt) % 81
            raw[off + 25] = (r * 5 + b * 19 + salt * 3) % 81
            scale = 0.25 + 0.0625 * ((r + b + salt) % 8)
            raw[off + 26 : off + 28] = struct.pack("<e", scale)
    return memoryview(raw)


def bf16_weights(rows: int, n: int, salt: int) -> memoryview:
    vals = (0x3F80, 0xBF00, 0x3E80, 0xBE80, 0x3F00, 0xBF80)
    raw = bytearray(rows * n * 2)
    for r in range(rows):
        for i in range(n):
            bits = vals[(r * 3 + i + salt) % len(vals)]
            struct.pack_into("<H", raw, 2 * (r * n + i), bits)
    return memoryview(raw)


def meta(name: str, kind: str, rows: int, n: int = HIDDEN) -> dict:
    return {"name": name, "type_name": kind, "shape": [n, rows]}


def repeated_ptq1_weights(rows: int, n: int, salt: int) -> memoryview:
    blocks = n // 128
    row = bytearray(blocks * 28)
    for b in range(blocks):
        off = b * 28
        for j in range(24):
            row[off + j] = (b * 29 + j * 11 + salt) % 243
        row[off + 24] = (b * 13 + salt) % 81
        row[off + 25] = (b * 19 + salt * 3) % 81
        scale = 0.25 + 0.0625 * ((b + salt) % 8)
        row[off + 26 : off + 28] = struct.pack("<e", scale)
    return memoryview(bytearray(row * rows))


def repeated_bf16_weights(rows: int, n: int, salt: int) -> memoryview:
    vals = (0x3F80, 0xBF00, 0x3E80, 0xBE80, 0x3F00, 0xBF80)
    row = bytearray(n * 2)
    for i in range(n):
        struct.pack_into("<H", row, 2 * i, vals[(i + salt) % len(vals)])
    return memoryview(bytearray(row * rows))


def real_geometry_benchmark(library: Path) -> None:
    qkv_name = "blk.0.attn_qkv.weight"
    gate_name = "blk.0.attn_gate.weight"
    metadata = {
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 1024,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.weight_names": [qkv_name, gate_name],
        "prism.hadamard.inverse_weight_names": [],
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.sign_widths": [REAL_HIDDEN],
        "prism.hadamard.sign_values": [1] * REAL_HIDDEN,
        "prism.hadamard.gdn_v_grouped": False,
    }
    runtime = Bonsai2NativeRuntime(
        library,
        metadata,
        threads=4,
        max_rows=max(
            REAL_HIDDEN,
            REAL_QKV_ROWS + REAL_GATE_ROWS + 32,
        ),
    )
    try:
        x = array(
            "f",
            (
                0.25 * math.sin(i * 0.003125)
                + 0.125 * math.cos(i * 0.0015625)
                for i in range(REAL_HIDDEN)
            ),
        )
        qkv_w = repeated_ptq1_weights(
            REAL_QKV_ROWS, REAL_HIDDEN, 23
        )
        gate_w = repeated_ptq1_weights(
            REAL_GATE_ROWS, REAL_HIDDEN, 31
        )
        beta_w = repeated_bf16_weights(
            REAL_SMALL_ROWS, REAL_HIDDEN, 37
        )
        alpha_w = repeated_bf16_weights(
            REAL_SMALL_ROWS, REAL_HIDDEN, 41
        )
        metas = {
            "qkv": meta(
                qkv_name, "PTQ1_0", REAL_QKV_ROWS, REAL_HIDDEN
            ),
            "gate": meta(
                gate_name, "PTQ1_0", REAL_GATE_ROWS, REAL_HIDDEN
            ),
            "beta": meta(
                "blk.0.ssm_beta.weight",
                "BF16",
                REAL_SMALL_ROWS,
                REAL_HIDDEN,
            ),
            "alpha": meta(
                "blk.0.ssm_alpha.weight",
                "BF16",
                REAL_SMALL_ROWS,
                REAL_HIDDEN,
            ),
        }

        reference = baseline(
            runtime, x, qkv_w, gate_w, beta_w, alpha_w, metas
        )
        candidate = runtime.recurrent_projections(
            x,
            qkv_w, metas["qkv"],
            gate_w, metas["gate"],
            beta_w, metas["beta"],
            alpha_w, metas["alpha"],
        )
        for label, ref_values, got_values in zip(
            ("qkv", "gate", "beta", "alpha"),
            reference,
            candidate,
        ):
            if f32_bytes(ref_values) != f32_bytes(got_values):
                raise AssertionError(
                    f"real-geometry recurrent bundle {label} "
                    "bitwise mismatch"
                )

        def run_baseline():
            return baseline(
                runtime, x, qkv_w, gate_w, beta_w, alpha_w, metas
            )

        def run_bundle():
            return runtime.recurrent_projections(
                x,
                qkv_w, metas["qkv"],
                gate_w, metas["gate"],
                beta_w, metas["beta"],
                alpha_w, metas["alpha"],
            )

        # Warm both paths once, then alternate order so CPU frequency and
        # runner noise do not systematically favor either implementation.
        run_baseline()
        run_bundle()
        baseline_samples = []
        bundle_samples = []
        for first, second in (
            (run_baseline, run_bundle),
            (run_bundle, run_baseline),
            (run_baseline, run_bundle),
            (run_bundle, run_baseline),
        ):
            t0 = time.perf_counter()
            first()
            t1 = time.perf_counter()
            second()
            t2 = time.perf_counter()
            if first is run_baseline:
                baseline_samples.append(t1 - t0)
                bundle_samples.append(t2 - t1)
            else:
                bundle_samples.append(t1 - t0)
                baseline_samples.append(t2 - t1)

        baseline_samples.sort()
        bundle_samples.sort()
        baseline_median = 0.5 * (
            baseline_samples[1] + baseline_samples[2]
        )
        bundle_median = 0.5 * (
            bundle_samples[1] + bundle_samples[2]
        )
        speedup = (
            baseline_median / bundle_median
            if bundle_median else float("inf")
        )
        print(
            "QWEN38_BONSAI2_RECURRENT_PROJECTIONS_REAL_GEOMETRY_PASS "
            f"hidden={REAL_HIDDEN} qkv_rows={REAL_QKV_ROWS} "
            f"gate_rows={REAL_GATE_ROWS} small_rows={REAL_SMALL_ROWS} "
            "threads=4 "
            f"baseline_median_seconds={baseline_median:.6f} "
            f"bundle_median_seconds={bundle_median:.6f} "
            f"speedup={speedup:.4f}x "
            f"baseline_samples={baseline_samples!r} "
            f"bundle_samples={bundle_samples!r}"
        )
    finally:
        runtime.close()


def f32_bytes(values) -> bytes:
    return array("f", values).tobytes()


def baseline(runtime, x, qkv_w, gate_w, beta_w, alpha_w, metas):
    prepared = runtime.prepare_activation(metas["qkv"]["name"], x)
    qkv = runtime.matvec_prepared(qkv_w, metas["qkv"], prepared)
    gate = runtime.matvec_prepared(gate_w, metas["gate"], prepared)
    beta = runtime.matvec(beta_w, metas["beta"], x)
    alpha = runtime.matvec(alpha_w, metas["alpha"], x)
    return qkv, gate, beta, alpha


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_bonsai2_recurrent_projections.py LIB")

    qkv_name = "blk.0.attn_qkv.weight"
    gate_name = "blk.0.attn_gate.weight"
    metadata = {
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 128,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.weight_names": [qkv_name, gate_name],
        "prism.hadamard.inverse_weight_names": [],
        "prism.hadamard.sign_mode": "identity",
        "prism.hadamard.gdn_v_grouped": False,
    }
    runtime = Bonsai2NativeRuntime(
        Path(sys.argv[1]),
        metadata,
        threads=2,
        max_rows=max(HIDDEN, QKV_ROWS + GATE_ROWS + 32),
    )
    try:
        x = array(
            "f",
            (
                0.25 * math.sin(i * 0.03125)
                + 0.125 * math.cos(i * 0.015625)
                for i in range(HIDDEN)
            ),
        )
        qkv_w = ptq1_weights(QKV_ROWS, HIDDEN, 3)
        gate_w = ptq1_weights(GATE_ROWS, HIDDEN, 7)
        beta_w = bf16_weights(SMALL_ROWS, HIDDEN, 11)
        alpha_w = bf16_weights(SMALL_ROWS, HIDDEN, 17)
        metas = {
            "qkv": meta(qkv_name, "PTQ1_0", QKV_ROWS),
            "gate": meta(gate_name, "PTQ1_0", GATE_ROWS),
            "beta": meta("blk.0.ssm_beta.weight", "BF16", SMALL_ROWS),
            "alpha": meta("blk.0.ssm_alpha.weight", "BF16", SMALL_ROWS),
        }

        reference = baseline(
            runtime, x, qkv_w, gate_w, beta_w, alpha_w, metas
        )
        candidate = runtime.recurrent_projections(
            x,
            qkv_w, metas["qkv"],
            gate_w, metas["gate"],
            beta_w, metas["beta"],
            alpha_w, metas["alpha"],
        )
        labels = ("qkv", "gate", "beta", "alpha")
        for label, ref_values, got_values in zip(labels, reference, candidate):
            if f32_bytes(ref_values) != f32_bytes(got_values):
                raise AssertionError(
                    f"recurrent projection bundle {label} bitwise mismatch"
                )

        reps = 12
        t0 = time.perf_counter()
        for _ in range(reps):
            baseline(runtime, x, qkv_w, gate_w, beta_w, alpha_w, metas)
        baseline_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(reps):
            runtime.recurrent_projections(
                x,
                qkv_w, metas["qkv"],
                gate_w, metas["gate"],
                beta_w, metas["beta"],
                alpha_w, metas["alpha"],
            )
        bundle_s = time.perf_counter() - t0
        speedup = baseline_s / bundle_s if bundle_s else float("inf")
        print(
            "QWEN38_BONSAI2_RECURRENT_PROJECTIONS_BITWISE_PASS "
            f"baseline_seconds={baseline_s:.6f} "
            f"bundle_seconds={bundle_s:.6f} speedup={speedup:.4f}x"
        )
    finally:
        runtime.close()

    real_geometry_benchmark(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
