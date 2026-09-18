#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime
import bonsai2_two_token as t2

F16_ONE = bytes((0x00, 0x3C))
BF16_ONE = bytes((0x80, 0x3F))


def metadata():
    return {
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 128,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "identity",
        "prism.hadamard.weight_names": ["blk.0.ptq.weight", "blk.0.pq.weight"],
        "prism.hadamard.inverse_weight_names": [],
        "prism.hadamard.gdn_v_grouped": False,
    }


def run_quant_profile(lib: Path) -> None:
    rt = Bonsai2NativeRuntime(lib, metadata(), threads=2, max_rows=4)
    try:
        x = [((i % 17) - 8) / 16.0 for i in range(128)]

        ptq = bytearray(28)
        ptq[26:28] = F16_ONE
        out = rt.matvec(
            memoryview(ptq),
            {"name": "blk.0.ptq.weight", "type_name": "PTQ1_0", "shape": [128, 1]},
            x,
        )
        assert len(out) == 1

        pq = bytearray(34)
        pq[0:2] = F16_ONE
        out = rt.matvec(
            memoryview(pq),
            {"name": "blk.0.pq.weight", "type_name": "PQ2_0", "shape": [128, 1]},
            x,
        )
        assert len(out) == 1

        bf = bytearray(BF16_ONE * 128)
        out = rt.matvec(
            memoryview(bf),
            {"name": "blk.0.bf.weight", "type_name": "BF16", "shape": [128, 1]},
            x,
        )
        assert len(out) == 1

        row = rt.dequantize_lookup_row(ptq, "PTQ1_0", 128, "token_embd.weight")
        assert len(row) == 128

        report = rt.report()
        timing = report["timing_seconds"]
        calls = report["timing_calls"]
        for key in (
            "hadamard",
            "q8_quantize",
            "ptq1_matvec",
            "pq2_matvec",
            "bf16_matvec",
            "lookup_dequantize",
        ):
            assert key in timing, (key, report)
            assert timing[key] > 0.0, (key, timing[key])
            assert calls[key] > 0, (key, calls[key])
    finally:
        rt.close()


def run_state_profile(lib: Path) -> None:
    rt = t2.GDNStateRuntime(lib, 2)
    try:
        state = (ctypes.c_float * t2.STATE_ELEMS)()
        vec = (ctypes.c_float * (48 * 128))()
        gate = (ctypes.c_float * 48)(*([-0.1] * 48))
        beta = (ctypes.c_float * 48)(*([0.5] * 48))
        out = (ctypes.c_float * (48 * 128))()
        rc = rt.step(state, vec, vec, vec, gate, beta, out)
        assert rc == 0
        report = rt.report()
        assert report["timing_seconds"]["step"] > 0.0, report
        assert report["timing_calls"]["step"] == 1, report
    finally:
        rt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-lib", type=Path, required=True)
    ap.add_argument("--state-lib", type=Path, required=True)
    args = ap.parse_args()
    run_quant_profile(args.quant_lib)
    run_state_profile(args.state_lib)
    print("QWEN38_BONSAI2_RUNTIME_PROFILE_PASS")


if __name__ == "__main__":
    main()
