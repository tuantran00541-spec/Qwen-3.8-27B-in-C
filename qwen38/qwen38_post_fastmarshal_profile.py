#!/usr/bin/env python3
"""Clean exact hotspot profile after bulk matvec output marshaling.

Runs the proven current-best 11-token staged prefill path once.  Decoder
arithmetic is unchanged: Q8 noalloc, bulk output marshaling, native F32,
attention, GDN conv/output gate/repeat-scale, SwiGLU, RMSNorm, residual add,
and attention sigmoid gate all remain enabled.  This wrapper only times
matvec_many/prepare_many calls and verifies the established hidden/state/K3
anchors so the next optimization is selected from clean evidence rather than
cProfile-distorted timings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_gdn_repeat_scale_exact_probe as repeat_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from gdn_repeat_scale_runtime import ExactGDNRepeatScale
from native_f32_runtime import enable_native_f32
from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime
from quant_many_runtime import load_many_lib

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_MANY = {"many_calls": 400, "many_vectors": 4400, "many_rows": 42893312}
EXPECTED_REPEAT = {
    "calls": 48,
    "rows": 48 * len(PROMPT_IDS),
    "q_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
    "k_values": 48 * len(PROMPT_IDS) * gdn.VALUE_DIM,
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def tensor_role(name: str) -> str:
    for suffix in (
        "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
        "ssm_out.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ):
        if name.endswith(suffix):
            return suffix
    return name.rsplit(".", 1)[-1] if name else "unknown"


class TimingProfile:
    def __init__(self) -> None:
        self.role_seconds: dict[str, float] = defaultdict(float)
        self.role_calls: dict[str, int] = defaultdict(int)
        self.type_seconds: dict[str, float] = defaultdict(float)
        self.type_calls: dict[str, int] = defaultdict(int)
        self.prepare_seconds: dict[str, float] = defaultdict(float)
        self.prepare_calls: dict[str, int] = defaultdict(int)
        self.external_prepare_seconds: dict[str, float] = defaultdict(float)
        self.external_prepare_calls: dict[str, int] = defaultdict(int)
        self._matvec_depth = 0

    def record_prepare(self, kind: str, seconds: float) -> None:
        self.prepare_seconds[kind] += seconds
        self.prepare_calls[kind] += 1
        if self._matvec_depth == 0:
            self.external_prepare_seconds[kind] += seconds
            self.external_prepare_calls[kind] += 1

    def record_matvec(self, meta: dict[str, Any], seconds: float) -> None:
        role = tensor_role(str(meta.get("name", "")))
        kind = str(meta.get("type_name", "unknown"))
        self.role_seconds[role] += seconds
        self.role_calls[role] += 1
        self.type_seconds[kind] += seconds
        self.type_calls[kind] += 1

    def report(self) -> dict[str, Any]:
        return {
            "matvec_role_seconds": dict(sorted(self.role_seconds.items(), key=lambda kv: kv[1], reverse=True)),
            "matvec_role_calls": dict(self.role_calls),
            "matvec_type_seconds": dict(sorted(self.type_seconds.items(), key=lambda kv: kv[1], reverse=True)),
            "matvec_type_calls": dict(self.type_calls),
            "prepare_seconds_by_kind": dict(self.prepare_seconds),
            "prepare_calls_by_kind": dict(self.prepare_calls),
            "external_prepare_seconds_by_kind": dict(self.external_prepare_seconds),
            "external_prepare_calls_by_kind": dict(self.external_prepare_calls),
        }


def sanity() -> None:
    if tensor_role("blk.7.ffn_down.weight") != "ffn_down.weight":
        raise SystemExit("tensor-role sanity failed")
    if EXPECTED_MANY["many_rows"] != 42893312:
        raise SystemExit("quant-many coverage sanity failed")
    if EXPECTED_REPEAT["rows"] != 528:
        raise SystemExit("repeat coverage sanity failed")
    print("QWEN38_POST_FASTMARSHAL_PROFILE_SANITY PASS")


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        base_runtime = engine.runtime
        many = FastMarshalQuantManyRuntime(base_runtime, load_many_lib(args.many_lib))
        engine.runtime = many
        timing = TimingProfile()

        original_many = many.matvec_many
        original_prepare = many.prepare_many

        def timed_prepare(xs, kind):
            t0 = time.monotonic()
            out = original_prepare(xs, kind)
            timing.record_prepare(str(kind), time.monotonic() - t0)
            return out

        def timed_many(weights, meta, xs, prepared=None):
            timing._matvec_depth += 1
            t0 = time.monotonic()
            try:
                return original_many(weights, meta, xs, prepared=prepared)
            finally:
                seconds = time.monotonic() - t0
                timing._matvec_depth -= 1
                timing.record_matvec(meta, seconds)

        many.prepare_many = timed_prepare
        many.matvec_many = timed_many
        repeat_core = ExactGDNRepeatScale(args.repeat_lib)
        try:
            (
                hidden, prefill_seconds, k3_bytes, native_stats, rms_report,
                residual_report, attn_gate_report, repeat_report,
            ) = repeat_probe._current_best_run(engine, args, repeat_core=repeat_core)
        finally:
            many.prepare_many = original_prepare
            many.matvec_many = original_many

        hidden_sha = block._digest_hidden_rows(hidden)
        state_sha = pair.snapshot_digest(pair.capture_state(engine))
        if hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"post-fastmarshal hidden anchor changed: {hidden_sha}")
        if state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"post-fastmarshal state anchor changed: {state_sha}")
        if k3_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"post-fastmarshal expected one K3 stream, got {k3_bytes}")
        if many.report() != EXPECTED_MANY:
            raise RuntimeError(f"unexpected quant-many coverage: {many.report()}")
        if repeat_report != EXPECTED_REPEAT:
            raise RuntimeError(f"unexpected repeat coverage: {repeat_report}")
        if rms_report.get("total_rows") != 6336:
            raise RuntimeError(f"unexpected RMSNorm coverage: {rms_report}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("post-fastmarshal profile requires direct I/O")

        profile = timing.report()
        matvec_seconds = sum(timing.role_seconds.values())
        external_prepare_seconds = sum(timing.external_prepare_seconds.values())
        native_helper_seconds = sum(float(v) for v in native_stats.values())
        accounted_seconds = matvec_seconds + external_prepare_seconds + native_helper_seconds
        unprofiled_seconds = prefill_seconds - accounted_seconds
        if unprofiled_seconds < -0.05:
            raise RuntimeError(
                f"post-fastmarshal accounting became negative: {unprofiled_seconds}")

        payload = {
            "schema": "qwen38-post-fastmarshal-profile-v1",
            "status": "PASS",
            "claim": "clean exact post-fastmarshal hotspot profile on current-best 11-token path",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_sha256": hidden_sha,
            "state_sha256": state_sha,
            "prefill_seconds": prefill_seconds,
            "k3_bytes": k3_bytes,
            "matvec_seconds_inclusive": matvec_seconds,
            "external_prepare_seconds": external_prepare_seconds,
            "native_helper_seconds": native_helper_seconds,
            "unprofiled_python_orchestration_seconds": unprofiled_seconds,
            "profile": profile,
            "native_seconds": native_stats,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "fast_marshal": {
                "calls": many.many_calls,
                "values": many.many_rows,
            },
            "rmsnorm": rms_report,
            "residual": residual_report,
            "attention_gate": attn_gate_report,
            "gdn_repeat_scale": repeat_report,
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "accounting_note": (
                "matvec_seconds includes any activation preparation performed inside matvec_many; "
                "external_prepare_seconds counts only prepare_many calls outside matvec_many"
            ),
            "baseline_fastmarshal_ab_run": 33840839301,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_POST_FASTMARSHAL_PROFILE_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
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
        sanity()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
