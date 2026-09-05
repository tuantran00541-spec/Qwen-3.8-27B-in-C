#!/usr/bin/env python3
"""Exact activation-quantization input marshaling A/B on current-best Qwen3.8.

Reference and candidate both keep the proven fast C->Python matvec output
marshal, Q8 noalloc bridge, native GDN repeat-scale, and every other current-best
exact helper.  Candidate changes only Python->F32 input construction before the
same native Q8_K/Q8_0 activation quantizer functions.
"""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import resource
import struct
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_gdn_repeat_scale_exact_probe as repeat_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from fast_quantize_runtime import FastActivationQuantizer, find_base_quant_runtime
from gdn_repeat_scale_runtime import ExactGDNRepeatScale
from native_f32_runtime import enable_native_f32
from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime
from quant_many_runtime import load_many_lib

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_MANY = {"many_calls": 400, "many_vectors": 4400, "many_rows": 42893312}
EXPECTED_RESIDUAL = {"calls": 128, "rows": 1408, "values": 7208960}
EXPECTED_ATTN_GATE = {"calls": 176, "values": 1081344}
EXPECTED_REPEAT = {
    "calls": 48,
    "rows": 48 * len(PROMPT_IDS),
    "q_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "k_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
}
EXPECTED_FAST_QUANT_CALLS = 3696


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", map(float, values)).tobytes()


def _finite_f32(bits: int) -> float:
    v = struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("fixture requested non-finite F32")
    return v


def _fixture(n: int, salt: int) -> list[float]:
    edge_bits = (
        0x00000000, 0x80000000, 0x00000001, 0x007FFFFF,
        0x00800000, 0x3F800000, 0xBF800000, 0x3EAAAAAB,
        0x41200000, 0xC1200000,
    )
    out: list[float] = []
    for i in range(n):
        if i < len(edge_bits):
            out.append(_finite_f32(edge_bits[(i + salt) % len(edge_bits)]))
        else:
            out.append((((i * 104729 + salt * 1543) % 200003) - 100001) / 4096.0)
    return out


def sanity(quant_lib: Path) -> None:
    runtime = gdn.QuantRuntime(gdn._load_native(quant_lib))
    fast = FastActivationQuantizer(runtime)
    cases = (
        ("Q8_0", 32), ("Q8_0", 5120), ("Q8_0", 6144), ("Q8_0", 17408),
        ("Q6_K", 256), ("Q6_K", 5120), ("Q6_K", 6144), ("Q6_K", 17408),
    )
    for salt, (kind, n) in enumerate(cases, 1):
        x = _fixture(n, salt)
        ref_buf, ref_n = fast._original_quantize(x, kind)
        cand_buf, cand_n = fast.quantize(x, kind)
        if ref_n != cand_n or bytes(ref_buf) != bytes(cand_buf):
            raise SystemExit(f"fast quantize byte mismatch kind={kind} n={n}")
    print("QWEN38_FAST_QUANTIZE_EXACT_SANITY PASS")


def _run_current_best(engine, args, repeat_core: ExactGDNRepeatScale):
    return repeat_probe._current_best_run(engine, args, repeat_core=repeat_core)


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    fast_quant: FastActivationQuantizer | None = None
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        base_runtime = engine.runtime
        base_quant = find_base_quant_runtime(base_runtime)
        initial = pair.capture_state(engine)

        ref_many = FastMarshalQuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = ref_many
        ref_repeat = ExactGDNRepeatScale(args.repeat_lib)
        (
            ref_hidden, ref_seconds, ref_bytes, ref_stats, ref_rms,
            ref_residual, ref_attn_gate, ref_repeat_report,
        ) = _run_current_best(engine, args, ref_repeat)
        ref_final = pair.capture_state(engine)
        ref_hidden_sha = block._digest_hidden_rows(ref_hidden)
        ref_state_sha = pair.snapshot_digest(ref_final)
        ref_many_report = ref_many.report()

        pair.restore_state(engine, initial)
        fast_quant = FastActivationQuantizer(base_quant)
        fast_quant.install()
        cand_many = FastMarshalQuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = cand_many
        cand_repeat = ExactGDNRepeatScale(args.repeat_lib)
        (
            cand_hidden, cand_seconds, cand_bytes, cand_stats, cand_rms,
            cand_residual, cand_attn_gate, cand_repeat_report,
        ) = _run_current_best(engine, args, cand_repeat)
        cand_final = pair.capture_state(engine)
        cand_hidden_sha = block._digest_hidden_rows(cand_hidden)
        cand_state_sha = pair.snapshot_digest(cand_final)
        cand_many_report = cand_many.report()
        fast_quant_report = fast_quant.report()
        fast_quant.restore()
        fast_quant = None

        hidden_exact = len(ref_hidden) == len(cand_hidden) and all(
            _f32_bytes(a) == _f32_bytes(b) for a, b in zip(ref_hidden, cand_hidden)
        )
        state_exact, state_mismatch = pair.compare_current_to_snapshot(engine, ref_final)
        if not hidden_exact:
            raise RuntimeError("fast-quantize candidate hidden vectors are not bitwise exact")
        if not state_exact:
            raise RuntimeError(f"fast-quantize candidate state mismatch: {state_mismatch}")
        if ref_hidden_sha != KNOWN_HIDDEN_SHA256 or cand_hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError("known hidden anchor changed")
        if ref_state_sha != KNOWN_STATE_SHA256 or cand_state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError("known state anchor changed")
        if ref_bytes != K3_STREAM_BYTES or cand_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected K3 bytes ref={ref_bytes} cand={cand_bytes}")
        if ref_many_report != EXPECTED_MANY or cand_many_report != EXPECTED_MANY:
            raise RuntimeError(
                f"unexpected quant-many coverage ref={ref_many_report} cand={cand_many_report}")
        if ref_residual != EXPECTED_RESIDUAL or cand_residual != EXPECTED_RESIDUAL:
            raise RuntimeError(
                f"unexpected residual coverage ref={ref_residual} cand={cand_residual}")
        if ref_attn_gate != EXPECTED_ATTN_GATE or cand_attn_gate != EXPECTED_ATTN_GATE:
            raise RuntimeError(
                f"unexpected attention-gate coverage ref={ref_attn_gate} cand={cand_attn_gate}")
        if ref_repeat_report != EXPECTED_REPEAT or cand_repeat_report != EXPECTED_REPEAT:
            raise RuntimeError(
                f"unexpected repeat coverage ref={ref_repeat_report} cand={cand_repeat_report}")
        if ref_rms.get("total_rows") != 6336 or cand_rms.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected RMSNorm coverage ref={ref_rms} cand={cand_rms}")
        if int(fast_quant_report["calls"]) != EXPECTED_FAST_QUANT_CALLS:
            raise RuntimeError(f"unexpected fast-quantize coverage: {fast_quant_report}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("fast-quantize A/B requires direct I/O")

        payload = {
            "schema": "qwen38-fast-quantize-exact-ab-v1",
            "status": "PASS",
            "claim": "exact activation-quantizer input marshaling; same-run real GGUF A/B on fastmarshal current-best path",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_vectors_bitwise_exact": hidden_exact,
            "persistent_state_bitwise_exact": state_exact,
            "state_mismatch": state_mismatch,
            "hidden_sha256": cand_hidden_sha,
            "state_sha256": cand_state_sha,
            "reference_seconds_same_run": ref_seconds,
            "candidate_seconds_same_run": cand_seconds,
            "speedup_same_run": ref_seconds / cand_seconds if cand_seconds else None,
            "reference_k3_bytes": ref_bytes,
            "candidate_k3_bytes": cand_bytes,
            "reference_native_seconds": ref_stats,
            "candidate_native_seconds": cand_stats,
            "reference_quant_many": ref_many_report,
            "candidate_quant_many": cand_many_report,
            "fast_quantize": fast_quant_report,
            "reference_rmsnorm": ref_rms,
            "candidate_rmsnorm": cand_rms,
            "reference_residual": ref_residual,
            "candidate_residual": cand_residual,
            "reference_attention_gate": ref_attn_gate,
            "candidate_attention_gate": cand_attn_gate,
            "reference_gdn_repeat_scale": ref_repeat_report,
            "candidate_gdn_repeat_scale": cand_repeat_report,
            "native_f32": native_f32.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "activation_quantize_calls_per_prefill": EXPECTED_FAST_QUANT_CALLS,
                "reference": "ctypes c_float array construction via Python argument splat",
                "candidate": "array('f') construction plus zero-copy ctypes from_buffer view",
                "native_quantizer_change": False,
                "quantized_output_change": False,
                "arithmetic_change": False,
            },
            "baseline_fastmarshal_ab_run": 33840839301,
            "profile_evidence_run": 33842519940,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_FAST_QUANTIZE_REAL_BITWISE_PASS")
        return payload
    finally:
        if fast_quant is not None:
            fast_quant.restore()
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--quant-lib", type=Path, required=True)
    r = sub.add_parser("real")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "f32-lib", "attn-lib",
        "conv-lib", "gate-lib", "swiglu-lib", "rmsnorm-lib", "residual-lib",
        "attention-gate-lib", "repeat-lib", "inventory", "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity(args.quant_lib)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
