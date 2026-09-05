#!/usr/bin/env python3
"""Zero-model correctness checks for Q6_K/Q8_0 reference decoding."""
from __future__ import annotations

import json
import math
import struct

from gguf_quant_ref import (
    BLOCK_Q6_K,
    BLOCK_Q8_0,
    QK_Q6_K,
    QK8_0,
    dequantize_q6_k,
    dequantize_q8_0,
    matvec_rows,
    row_nbytes,
)


def pack_q8_block(scale: float, quants: list[int]) -> bytes:
    assert len(quants) == 32 and all(-128 <= q <= 127 for q in quants)
    return struct.pack("<e32b", scale, *quants)


def pack_q6_block(scale: float, scales: list[int], quants: list[int]) -> bytes:
    """Independent inverse-layout fixture encoder for one ggml block_q6_K."""
    assert len(scales) == 16 and all(-128 <= s <= 127 for s in scales)
    assert len(quants) == 256 and all(-32 <= q <= 31 for q in quants)
    values = [q + 32 for q in quants]
    ql = bytearray(128)
    qh = bytearray(64)
    for n in (0, 128):
        ql_base = 0 if n == 0 else 64
        qh_base = 0 if n == 0 else 32
        for l in range(32):
            q1 = values[n + l]
            q2 = values[n + l + 32]
            q3 = values[n + l + 64]
            q4 = values[n + l + 96]
            ql[ql_base + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4)
            ql[ql_base + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4)
            qh[qh_base + l] = (
                ((q1 >> 4) & 0x03)
                | (((q2 >> 4) & 0x03) << 2)
                | (((q3 >> 4) & 0x03) << 4)
                | (((q4 >> 4) & 0x03) << 6)
            )
    block = bytes(ql) + bytes(qh) + struct.pack("<16b", *scales) + struct.pack("<e", scale)
    assert len(block) == BLOCK_Q6_K
    return block


def expected_q6(scale: float, scales: list[int], quants: list[int]) -> list[float]:
    return [float(scale) * scales[i // 16] * quants[i] for i in range(256)]


def max_error(actual: list[float], expected: list[float]) -> float:
    assert len(actual) == len(expected)
    return max((abs(a - b) for a, b in zip(actual, expected)), default=0.0)


def main() -> None:
    assert row_nbytes("Q8_0", 32) == BLOCK_Q8_0 == 34
    assert row_nbytes("Q6_K", 256) == BLOCK_Q6_K == 210

    q8_a = list(range(-16, 16))
    q8_b = [31 - i for i in range(32)]
    raw_q8_a = pack_q8_block(0.5, q8_a)
    raw_q8_b = pack_q8_block(-0.25, q8_b)
    dec_q8_a = dequantize_q8_0(raw_q8_a, QK8_0)
    dec_q8_b = dequantize_q8_0(raw_q8_b, QK8_0)
    exp_q8_a = [0.5 * q for q in q8_a]
    exp_q8_b = [-0.25 * q for q in q8_b]
    q8_err = max(max_error(dec_q8_a, exp_q8_a), max_error(dec_q8_b, exp_q8_b))
    assert q8_err == 0.0

    scales_a = [-8, -5, -3, -1, 1, 2, 4, 7, -7, -4, -2, 1, 3, 5, 6, 8]
    scales_b = [8, 6, 5, 3, 1, -2, -4, -7, 7, 4, 2, -1, -3, -5, -6, -8]
    q6_a = [((i * 17 + 3) % 64) - 32 for i in range(256)]
    q6_b = [31 - ((i * 29 + 11) % 64) for i in range(256)]
    raw_q6_a = pack_q6_block(0.25, scales_a, q6_a)
    raw_q6_b = pack_q6_block(-0.125, scales_b, q6_b)
    dec_q6_a = dequantize_q6_k(raw_q6_a, QK_Q6_K)
    dec_q6_b = dequantize_q6_k(raw_q6_b, QK_Q6_K)
    exp_q6_a = expected_q6(0.25, scales_a, q6_a)
    exp_q6_b = expected_q6(-0.125, scales_b, q6_b)
    q6_err = max(max_error(dec_q6_a, exp_q6_a), max_error(dec_q6_b, exp_q6_b))
    assert q6_err == 0.0

    x_q6 = [((i % 13) - 6) / 8.0 for i in range(256)]
    got_q6_mv = matvec_rows(
        raw_q6_a + raw_q6_b,
        type_name="Q6_K",
        shape=(256, 2),
        vector=x_q6,
    )
    exp_q6_mv = [
        math.fsum(a * b for a, b in zip(exp_q6_a, x_q6)),
        math.fsum(a * b for a, b in zip(exp_q6_b, x_q6)),
    ]
    q6_mv_err = max_error(got_q6_mv, exp_q6_mv)
    assert q6_mv_err == 0.0

    x_q8 = [((i % 9) - 4) / 5.0 for i in range(32)]
    got_q8_mv = matvec_rows(
        raw_q8_a + raw_q8_b,
        type_name="Q8_0",
        shape=(32, 2),
        vector=x_q8,
    )
    exp_q8_mv = [
        math.fsum(a * b for a, b in zip(exp_q8_a, x_q8)),
        math.fsum(a * b for a, b in zip(exp_q8_b, x_q8)),
    ]
    q8_mv_err = max_error(got_q8_mv, exp_q8_mv)
    assert q8_mv_err == 0.0

    try:
        dequantize_q6_k(raw_q6_a[:-1], 256)
    except ValueError:
        truncated_rejected = True
    else:
        truncated_rejected = False
    assert truncated_rejected

    try:
        row_nbytes("Q8_0", 33)
    except ValueError:
        bad_width_rejected = True
    else:
        bad_width_rejected = False
    assert bad_width_rejected

    print(json.dumps({
        "schema": "qwen38-gguf-quant-ref-sanity-v1",
        "status": "PASS",
        "llama_cpp_reference_revision": "557614e0296ff4a5b6f649737a65ae2076eea2fd",
        "model_weights_downloaded": False,
        "q8_0_block_bytes": BLOCK_Q8_0,
        "q6_k_block_bytes": BLOCK_Q6_K,
        "q8_0_max_abs_error": q8_err,
        "q6_k_max_abs_error": q6_err,
        "q8_0_matvec_max_abs_error": q8_mv_err,
        "q6_k_matvec_max_abs_error": q6_mv_err,
        "truncated_rejected": truncated_rejected,
        "bad_width_rejected": bad_width_rejected,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
