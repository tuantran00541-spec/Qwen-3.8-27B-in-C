#!/usr/bin/env python3
from __future__ import annotations

from array import array
import ctypes
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

import bonsai2_two_token as t2
import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_ggml_exact as exact

KEY_DIM = gdn.KEY_DIM
VALUE_DIM = gdn.VALUE_DIM
CONV_DIM = gdn.CONV_DIM
V_HEADS = gdn.V_HEADS
HEAD_DIM = gdn.HEAD_DIM
SCALE_GDN = 1.0 / math.sqrt(HEAD_DIM)

FP = ctypes.POINTER(ctypes.c_float)
GDN_STEP = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    FP, FP, FP, FP, FP, FP, FP,
)

captured: dict[str, bytes] = {}


def f32(x: float) -> float:
    return exact.f32(x)


def rng_values(n: int, scale: float, seed: int) -> array:
    state = seed & 0xFFFFFFFF
    out = array("f")
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        raw = ((state >> 8) % 20001) - 10000
        out.append(f32(raw * (scale / 10000.0)))
    return out


def bytes_f32(values) -> bytes:
    return array("f", values).tobytes()


def ptr(buf: array) -> FP:
    return ctypes.cast((ctypes.c_float * len(buf)).from_buffer(buf), FP)


@GDN_STEP
def fake_state_step(pool, state, q, k, v, gate, beta, out):
    del pool, state
    captured["q"] = ctypes.string_at(q, VALUE_DIM * 4)
    captured["k"] = ctypes.string_at(k, VALUE_DIM * 4)
    captured["v"] = ctypes.string_at(v, VALUE_DIM * 4)
    captured["gate"] = ctypes.string_at(gate, V_HEADS * 4)
    captured["beta"] = ctypes.string_at(beta, V_HEADS * 4)
    ctypes.memmove(out, v, VALUE_DIM * 4)
    return 0


def reference_prepare(qkv: array, history: list[array], kernels: array,
                      alpha: array, beta_raw: array,
                      dt: array, a: array) -> tuple[array, array, array, array, array]:
    lib.qwen_bonsai2_gdn_conv_silu_f32.argtypes = [
        FP, FP, ctypes.c_size_t, FP, ctypes.c_size_t, FP
    ]
    lib.qwen_bonsai2_gdn_conv_silu_f32.restype = ctypes.c_int

    hist_flat = array("f")
    for row in history:
        hist_flat.extend(row)
    conv = array("f", [0.0]) * CONV_DIM
    rc = lib.qwen_bonsai2_gdn_conv_silu_f32(
        ptr(qkv),
        ptr(hist_flat) if hist_flat else None,
        len(history),
        ptr(kernels),
        CONV_DIM,
        ptr(conv),
    )
    if rc != 0:
        raise AssertionError(f"conv reference rc={rc}")

    beta = array("f", (exact.sigmoid_f32(v) for v in beta_raw))
    gate = array("f", (
        t2.mulf(a[i], t2.softplusf(t2.addf(alpha[i], dt[i])))
        for i in range(V_HEADS)
    ))

    q = conv[:KEY_DIM]
    k = conv[KEY_DIM:2 * KEY_DIM]
    v = conv[2 * KEY_DIM:]

    qn: list[float] = []
    kn: list[float] = []
    for h in range(gdn.K_HEADS):
        lo = h * HEAD_DIM
        hi = lo + HEAD_DIM
        qn.extend(gdn.l2_norm(q[lo:hi]))
        kn.extend(gdn.l2_norm(k[lo:hi]))

    q48 = array("f")
    k48 = array("f")
    for _ in range(V_HEADS // gdn.K_HEADS):
        q48.extend(t2.mulf(vv, SCALE_GDN) for vv in qn)
        k48.extend(f32(vv) for vv in kn)

    return q48, k48, array("f", v), gate, beta


def main() -> None:
    global lib
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_bonsai2_recurrent_mid.py LIB")
    lib = ctypes.CDLL(sys.argv[1])

    try:
        fn = lib.qwen_bonsai2_recurrent_mid_f32
    except AttributeError as exc:
        raise AssertionError("missing qwen_bonsai2_recurrent_mid_f32") from exc

    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, FP,
        FP, FP, FP, FP, ctypes.c_size_t,
        FP, FP, FP, FP, FP, FP,
        ctypes.c_float, ctypes.c_float, FP,
    ]
    fn.restype = ctypes.c_int

    qkv = rng_values(CONV_DIM, 0.15, 0x13550001)
    history = [
        rng_values(CONV_DIM, 0.12, 0x13550010 + i)
        for i in range(3)
    ]
    kernels = rng_values(CONV_DIM * 4, 0.08, 0x13550020)
    alpha = rng_values(V_HEADS, 0.5, 0x13550030)
    beta_raw = rng_values(V_HEADS, 0.5, 0x13550040)
    dt = rng_values(V_HEADS, 0.1, 0x13550050)
    a = array("f", (-0.01 - abs(v) for v in rng_values(V_HEADS, 0.08, 0x13550060)))
    z = rng_values(VALUE_DIM, 0.4, 0x13550070)
    norm = array("f", (0.75 + abs(v) for v in rng_values(HEAD_DIM, 0.3, 0x13550080)))
    state = rng_values(V_HEADS * HEAD_DIM * HEAD_DIM, 0.01, 0x13550090)

    qref, kref, vref, gateref, betaref = reference_prepare(
        qkv, history, kernels, alpha, beta_raw, dt, a
    )

    hist0, hist1, hist2 = history
    gated = array("f", [0.0]) * VALUE_DIM
    before_state = state.tobytes()
    rc = fn(
        None,
        ctypes.cast(fake_state_step, ctypes.c_void_p),
        ptr(state),
        ptr(qkv),
        ptr(hist0), ptr(hist1), ptr(hist2), 3,
        ptr(kernels), ptr(alpha), ptr(beta_raw), ptr(dt), ptr(a),
        ptr(z), ptr(norm),
        ctypes.c_float(gdn.RMS_EPS),
        ctypes.c_float(SCALE_GDN),
        ptr(gated),
    )
    if rc != 0:
        raise AssertionError(f"recurrent mid rc={rc}")
    if state.tobytes() != before_state:
        raise AssertionError("fake callback state unexpectedly changed")

    expected = {
        "q": qref.tobytes(),
        "k": kref.tobytes(),
        "v": vref.tobytes(),
        "gate": gateref.tobytes(),
        "beta": betaref.tobytes(),
    }
    for name, ref in expected.items():
        got = captured.get(name)
        if got != ref:
            raise AssertionError(
                f"{name} bitwise mismatch got={len(got or b'')} ref={len(ref)}"
            )

    lib.qwen_bonsai2_gdn_norm_gate_f32.argtypes = [
        FP, FP, FP, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, FP
    ]
    lib.qwen_bonsai2_gdn_norm_gate_f32.restype = ctypes.c_int
    gated_ref = array("f", [0.0]) * VALUE_DIM
    rc = lib.qwen_bonsai2_gdn_norm_gate_f32(
        ptr(vref), ptr(norm), ptr(z), V_HEADS, HEAD_DIM,
        ctypes.c_float(gdn.RMS_EPS), ptr(gated_ref)
    )
    if rc != 0:
        raise AssertionError(f"norm gate reference rc={rc}")
    if gated.tobytes() != gated_ref.tobytes():
        raise AssertionError("gated output bitwise mismatch")

    print("QWEN38_BONSAI2_RECURRENT_MID_BITWISE_PASS")


if __name__ == "__main__":
    main()
