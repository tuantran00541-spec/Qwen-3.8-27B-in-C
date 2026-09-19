#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime
import bonsai2_two_token as t2

HIDDEN = 256
INTERMEDIATE = 512
BLOCK = 128
FP = ctypes.POINTER(ctypes.c_float)
U8P = ctypes.POINTER(ctypes.c_uint8)
I8P = ctypes.POINTER(ctypes.c_int8)


def metadata() -> dict[str, object]:
    names = [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ]
    return {
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": BLOCK,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "identity",
        "prism.hadamard.weight_names": names,
        "prism.hadamard.inverse_weight_names": [],
        "prism.hadamard.gdn_v_grouped": False,
    }


class RNG:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def u32(self) -> int:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state

    def f32(self, scale: float) -> float:
        x = ((self.u32() >> 8) % 20001) - 10000
        return float(x) * (scale / 10000.0)


def ptq1_weights(rows: int, n: int, seed: int) -> bytearray:
    if n % 128:
        raise AssertionError(n)
    rng = RNG(seed)
    out = bytearray()
    scales = (0.125, 0.25, 0.5, 1.0)
    for _ in range(rows * (n // 128)):
        out.extend((rng.u32() & 0xFF) for _ in range(26))
        out.extend(struct.pack("<e", scales[rng.u32() % len(scales)]))
    return out


def meta(name: str, n: int, rows: int) -> dict[str, object]:
    return {
        "name": name,
        "type_name": "PTQ1_0",
        "shape": [n, rows],
    }



class DelegationProbeRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def ffn(
        self,
        x,
        gate_weights,
        gate_meta,
        up_weights,
        up_meta,
        down_weights,
        down_meta,
    ):
        self.calls += 1
        assert list(x) == [1.0, 2.0, 3.0]
        assert gate_weights == "gate"
        assert up_weights == "up"
        assert down_weights == "down"
        assert gate_meta["name"] == "blk.0.ffn_gate.weight"
        assert up_meta["name"] == "blk.0.ffn_up.weight"
        assert down_meta["name"] == "blk.0.ffn_down.weight"
        return [9.0, 8.0, 7.0]


def verify_t2_ffn_delegates_once() -> None:
    runtime = DelegationProbeRuntime()
    metas = {
        "blk.0.ffn_gate.weight": {"name": "blk.0.ffn_gate.weight"},
        "blk.0.ffn_up.weight": {"name": "blk.0.ffn_up.weight"},
        "blk.0.ffn_down.weight": {"name": "blk.0.ffn_down.weight"},
    }

    def view(name: str):
        return {
            "ffn_gate.weight": "gate",
            "ffn_up.weight": "up",
            "ffn_down.weight": "down",
        }[name]

    out = t2.ffn(runtime, view, metas, "blk.0", [1.0, 2.0, 3.0])
    if out != [9.0, 8.0, 7.0]:
        raise AssertionError(out)
    if runtime.calls != 1:
        raise AssertionError(f"expected one native FFN delegation, got {runtime.calls}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_bonsai2_native_ffn.py LIB")
    lib_path = Path(sys.argv[1])
    runtime = Bonsai2NativeRuntime(
        lib_path, metadata(), threads=2, max_rows=INTERMEDIATE
    )
    try:
        try:
            fn = runtime.lib.qwen_bonsai2_pool_ffn_ptq1_0
        except AttributeError as exc:
            raise AssertionError(
                "missing qwen_bonsai2_pool_ffn_ptq1_0"
            ) from exc

        fn.argtypes = [
            ctypes.c_void_p,
            FP,
            ctypes.c_size_t,
            ctypes.c_size_t,
            U8P,
            ctypes.c_size_t,
            U8P,
            ctypes.c_size_t,
            U8P,
            ctypes.c_size_t,
            ctypes.c_size_t,
            I8P,
            I8P,
            FP,
        ]
        fn.restype = ctypes.c_int

        gate_w = ptq1_weights(INTERMEDIATE, HIDDEN, 0x135501)
        up_w = ptq1_weights(INTERMEDIATE, HIDDEN, 0x135502)
        down_w = ptq1_weights(HIDDEN, INTERMEDIATE, 0x135503)
        rng = RNG(0x135504)
        x = [rng.f32(0.75) for _ in range(HIDDEN)]

        gate_meta = meta("blk.0.ffn_gate.weight", HIDDEN, INTERMEDIATE)
        up_meta = meta("blk.0.ffn_up.weight", HIDDEN, INTERMEDIATE)
        down_meta = meta("blk.0.ffn_down.weight", INTERMEDIATE, HIDDEN)

        prepared = runtime.prepare_activation(
            "blk.0.ffn_gate.weight", x
        )
        gate = runtime.matvec_prepared(
            memoryview(gate_w), gate_meta, prepared
        )
        up = runtime.matvec_prepared(
            memoryview(up_w), up_meta, prepared
        )
        sw = runtime.swiglu(gate, up)
        reference = runtime.matvec(
            memoryview(down_w), down_meta, sw
        )

        x_buf = (ctypes.c_float * HIDDEN)(*x)
        out = (ctypes.c_float * HIDDEN)()
        gate_buf = (ctypes.c_uint8 * len(gate_w)).from_buffer(gate_w)
        up_buf = (ctypes.c_uint8 * len(up_w)).from_buffer(up_w)
        down_buf = (ctypes.c_uint8 * len(down_w)).from_buffer(down_w)

        rc = fn(
            runtime.pool,
            x_buf,
            HIDDEN,
            INTERMEDIATE,
            gate_buf,
            len(gate_w),
            up_buf,
            len(up_w),
            down_buf,
            len(down_w),
            BLOCK,
            None,
            None,
            out,
        )
        if rc != 0:
            raise AssertionError(f"native FFN rc={rc}")

        got = bytes(memoryview(out).cast("B"))
        want_arr = (ctypes.c_float * HIDDEN)(*reference)
        want = bytes(memoryview(want_arr).cast("B"))
        if got != want:
            raise AssertionError("native FFN output is not bitwise identical")

        wrapped = runtime.ffn(
            x,
            memoryview(gate_w),
            gate_meta,
            memoryview(up_w),
            up_meta,
            memoryview(down_w),
            down_meta,
        )
        wrapped_arr = (ctypes.c_float * HIDDEN)(*wrapped)
        if bytes(memoryview(wrapped_arr).cast("B")) != want:
            raise AssertionError("runtime FFN wrapper is not bitwise identical")

        verify_t2_ffn_delegates_once()
        print("QWEN38_BONSAI2_NATIVE_FFN_BITWISE_PASS")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
