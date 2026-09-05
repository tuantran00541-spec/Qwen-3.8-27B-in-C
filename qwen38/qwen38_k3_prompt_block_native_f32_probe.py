#!/usr/bin/env python3
"""Real exact block-prefill probe for native F32 matrix arithmetic.

The proven layer-major scheduler and all quantized Q6_K/Q8_0 kernels stay
unchanged.  The candidate only replaces QuantRuntime's F32 matrix path
(Python f32_vector + math.fsum) with the strict native CPython-compatible fsum
kernel already double-bitwise gated synthetically.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_prefill_probe as block

KNOWN_PROMPT_IDS = [7734, 264, 220, 22, 15, 15, 36093, 8627, 383, 38896, 13]
KNOWN_HIDDEN_SHA256 = "e40dfb2d14456006608b095dd0c6bd018cdeed4214fdc573c8e352fb463f2e04"
KNOWN_STATE_SHA256 = "41f6fcd8f9947833956aaad0175da197456a3e678e0e31b40c5d7a08560fda06"
K3_STREAM_BYTES = 21_127_430_144


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def load_f32_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    fp = ctypes.POINTER(ctypes.c_float)
    dp = ctypes.POINTER(ctypes.c_double)
    lib.qwen_matvec_f32_fsum_exact.argtypes = [
        fp, ctypes.c_size_t, ctypes.c_size_t, dp, dp,
    ]
    lib.qwen_matvec_f32_fsum_exact.restype = ctypes.c_int
    return lib


class NativeF32Runtime:
    """Delegate quantized work unchanged; replace only F32 matrix matvec."""

    def __init__(self, runtime, f32_lib):
        self.runtime = runtime
        self.f32_lib = f32_lib
        self.native_f32_calls = 0
        self.native_f32_rows = 0
        self.native_f32_terms = 0

    def quantize(self, x: Sequence[float], kind: str):
        return self.runtime.quantize(x, kind)

    @property
    def activation_quantizations(self):
        return self.runtime.activation_quantizations

    @property
    def matvec_rows(self):
        return self.runtime.matvec_rows

    def matvec(self, weights: memoryview, meta: dict[str, Any], x: Sequence[float], prepared=None):
        if meta["type_name"] != "F32":
            return self.runtime.matvec(weights, meta, x, prepared=prepared)

        if sys.byteorder != "little":
            raise RuntimeError("native F32 probe requires little-endian host")
        ne0, rows = map(int, meta["shape"])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: x={len(x)} ne0={ne0}")
        expected_bytes = ne0 * rows * 4
        if len(weights) != expected_bytes:
            raise ValueError(
                f"{meta['name']}: F32 bytes={len(weights)} expected={expected_bytes}"
            )

        # K3 ring slots are mutable aligned byte buffers.  from_buffer keeps the
        # candidate zero-copy: the native kernel reads the exact bound tensor
        # bytes instead of decoding/copying the matrix into Python objects.
        w_arr = (ctypes.c_float * (ne0 * rows)).from_buffer(weights)
        x_arr = (ctypes.c_double * ne0)(*map(float, x))
        out = (ctypes.c_double * rows)()
        rc = self.f32_lib.qwen_matvec_f32_fsum_exact(w_arr, rows, ne0, x_arr, out)
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: native F32 fsum matvec rc={rc}")
        self.native_f32_calls += 1
        self.native_f32_rows += rows
        self.native_f32_terms += rows * ne0
        return [float(out[i]) for i in range(rows)]


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.native_lib, args.state_lib, args.inventory, args.work_dir,
    )
    started = time.monotonic()
    try:
        initial = pair.capture_state(engine)

        # A: proven exact layer-major block using the current Python F32 path.
        reader0 = int(engine.reader.report()["bytes_read"])
        a_started = time.monotonic()
        ref_hidden = block.step_block(engine, KNOWN_PROMPT_IDS)
        a_seconds = time.monotonic() - a_started
        a_bytes = int(engine.reader.report()["bytes_read"]) - reader0
        ref_state = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_state)
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or ref_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("proven block-v1 digest anchor changed")
        if a_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"block-v1 K3 bytes changed: {a_bytes}")

        # B: identical scheduler/state equations with only F32 matrix matvec
        # delegated to the strict native fsum implementation.
        pair.restore_state(engine, initial)
        base_runtime = engine.runtime
        native_runtime = NativeF32Runtime(base_runtime, load_f32_lib(args.f32_lib))
        engine.runtime = native_runtime
        reader1 = int(engine.reader.report()["bytes_read"])
        b_started = time.monotonic()
        cand_hidden = block.step_block(engine, KNOWN_PROMPT_IDS)
        b_seconds = time.monotonic() - b_started
        b_bytes = int(engine.reader.report()["bytes_read"]) - reader1
        cand_state = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_state)

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            block._f32_bytes(a) == block._f32_bytes(b)
            for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_state)
        if not hidden_exact or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("native-F32 block hidden vectors are not bitwise exact")
        if not state_exact or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"native-F32 block state mismatch: {state_mismatch}")
        if b_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"native-F32 block K3 bytes changed: {b_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("real native-F32 probe requires direct I/O")
        if native_runtime.native_f32_calls <= 0:
            raise RuntimeError("candidate did not execute any native F32 matvec")

        payload = {
            "schema": "qwen38-k3-prompt-block-native-f32-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": KNOWN_PROMPT_IDS,
            "prompt_token_count": len(KNOWN_PROMPT_IDS),
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "hidden_sha256": cand_hidden_sha,
            "state_sha256": cand_state_sha,
            "block_v1_seconds_same_run": a_seconds,
            "block_native_f32_seconds_same_run": b_seconds,
            "speedup_vs_block_v1_same_run": a_seconds / b_seconds,
            "block_v1_k3_bytes": a_bytes,
            "block_native_f32_k3_bytes": b_bytes,
            "native_f32_calls": native_runtime.native_f32_calls,
            "native_f32_rows": native_runtime.native_f32_rows,
            "native_f32_terms": native_runtime.native_f32_terms,
            "reader": reader,
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "block_v1_seconds_same_run": a_seconds,
            "block_native_f32_seconds_same_run": b_seconds,
            "speedup_vs_block_v1_same_run": payload["speedup_vs_block_v1_same_run"],
            "block_v1_k3_bytes": a_bytes,
            "block_native_f32_k3_bytes": b_bytes,
            "native_f32_calls": native_runtime.native_f32_calls,
            "native_f32_terms": native_runtime.native_f32_terms,
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_K3_PROMPT_BLOCK_NATIVE_F32_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert len(KNOWN_PROMPT_IDS) == 11
    assert K3_STREAM_BYTES == 21_127_430_144
    assert len(KNOWN_HIDDEN_SHA256) == 64
    assert len(KNOWN_STATE_SHA256) == 64
    print(json.dumps({
        "schema": "qwen38-k3-prompt-block-native-f32-sanity-v1",
        "status": "PASS",
        "candidate_delta": "F32 matrix matvec only",
        "quantized_path": "unchanged",
        "scheduler": "proven block.step_block unchanged",
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--native-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
