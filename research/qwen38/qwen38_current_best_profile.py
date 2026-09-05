#!/usr/bin/env python3
"""Clean exact hotspot profile for the current-best Qwen3.8 11-token prefill.

This is a measurement-only wrapper around the fully proven current-best stack:
fast activation-quantize marshaling, fast matvec output marshaling, Q8 noalloc,
GDN state batching, GDN/attention normal RMSNorm-many, attention Q/K head
RMSNorm-many, and all previously proven native exact helpers.

No decoder arithmetic is changed.  The profiler times quantized matvec_many and
prepare_many boundaries while preserving the established hidden/state/K3 and
direct-I/O invariants.  It is deliberately not a cProfile benchmark.
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
import qwen38_attention_head_rmsnorm_many_exact_probe as head_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from fast_quantize_runtime import FastActivationQuantizer, find_base_quant_runtime
from gdn_repeat_scale_runtime import ExactGDNRepeatScale
from gdn_state_batch_runtime import ExactGDNStateBatch
from native_f32_runtime import enable_native_f32
from quant_many_fastmarshal_runtime import FastMarshalQuantManyRuntime
from quant_many_runtime import load_many_lib
from rmsnorm_heads_many_runtime import ExactRMSNormHeadsMany
from rmsnorm_many_runtime import ExactRMSNormMany

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256
EXPECTED_MANY = dict(head_probe.EXPECTED_MANY)
EXPECTED_REPEAT = dict(head_probe.EXPECTED_REPEAT)
EXPECTED_BATCH = dict(head_probe.EXPECTED_BATCH)
EXPECTED_GDN_RMS_MANY = dict(head_probe.EXPECTED_GDN_RMS_MANY)
EXPECTED_ATTN_RMS_MANY = dict(head_probe.EXPECTED_ATTN_RMS_MANY)
EXPECTED_HEAD_MANY = dict(head_probe.EXPECTED_HEAD_MANY)
EXPECTED_ZERO_WRAPPER = dict(head_probe.EXPECTED_ZERO_WRAPPER)
EXPECTED_READER = dict(head_probe.EXPECTED_READER)
EXPECTED_FAST_QUANT_CALLS = int(head_probe.EXPECTED_FAST_QUANT_CALLS)
EXPECTED_FAST_QUANT_VALUES = 28_295_168
EXPECTED_TOTAL_RMS_ROWS = int(head_probe.EXPECTED_TOTAL_RMS_ROWS)
EXPECTED_TOTAL_RMS_VALUES = int(head_probe.EXPECTED_TOTAL_RMS_VALUES)


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
        self.prepare_seconds[kind] += float(seconds)
        self.prepare_calls[kind] += 1
        if self._matvec_depth == 0:
            self.external_prepare_seconds[kind] += float(seconds)
            self.external_prepare_calls[kind] += 1

    def record_matvec(self, meta: dict[str, Any], seconds: float) -> None:
        role = tensor_role(str(meta.get("name", "")))
        kind = str(meta.get("type_name", "unknown"))
        self.role_seconds[role] += float(seconds)
        self.role_calls[role] += 1
        self.type_seconds[kind] += float(seconds)
        self.type_calls[kind] += 1

    def report(self) -> dict[str, Any]:
        return {
            "matvec_role_seconds": dict(
                sorted(self.role_seconds.items(), key=lambda kv: kv[1], reverse=True)),
            "matvec_role_calls": dict(self.role_calls),
            "matvec_type_seconds": dict(
                sorted(self.type_seconds.items(), key=lambda kv: kv[1], reverse=True)),
            "matvec_type_calls": dict(self.type_calls),
            "prepare_seconds_by_kind": dict(self.prepare_seconds),
            "prepare_calls_by_kind": dict(self.prepare_calls),
            "external_prepare_seconds_by_kind": dict(self.external_prepare_seconds),
            "external_prepare_calls_by_kind": dict(self.external_prepare_calls),
        }


def sanity() -> None:
    if len(PROMPT_IDS) != 11:
        raise SystemExit(f"unexpected prompt token count: {len(PROMPT_IDS)}")
    if tensor_role("blk.63.ffn_down.weight") != "ffn_down.weight":
        raise SystemExit("tensor-role sanity failed")
    if EXPECTED_MANY != {"many_calls": 400, "many_vectors": 4400, "many_rows": 42893312}:
        raise SystemExit(f"unexpected quant-many contract: {EXPECTED_MANY}")
    if EXPECTED_HEAD_MANY["calls"] != 32 or EXPECTED_HEAD_MANY["head_rows"] != 4928:
        raise SystemExit(f"unexpected head RMSNorm-many contract: {EXPECTED_HEAD_MANY}")
    if (EXPECTED_TOTAL_RMS_ROWS, EXPECTED_TOTAL_RMS_VALUES) != (6336, 8470528):
        raise SystemExit("unexpected total RMSNorm contract")
    print("QWEN38_CURRENT_BEST_PROFILE_SANITY PASS")


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    fast: FastActivationQuantizer | None = None
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        base_runtime = engine.runtime
        base_quant = find_base_quant_runtime(base_runtime)

        fast = FastActivationQuantizer(base_quant)
        fast.install()
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
        state_batch_core = ExactGDNStateBatch(args.batch_state_lib)
        gdn_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        attention_rms_many = ExactRMSNormMany(args.rmsnorm_many_lib)
        head_many = ExactRMSNormHeadsMany(args.head_rmsnorm_many_lib)

        try:
            (
                hidden, prefill_seconds, k3_bytes, native_stats, rms_wrapper,
                residual_report, attn_gate_report, repeat_report,
            ) = head_probe._run_with_head_rms_many(
                engine, args, repeat_core, state_batch_core,
                gdn_rms_many, attention_rms_many, head_many)
        finally:
            many.prepare_many = original_prepare
            many.matvec_many = original_many

        hidden_sha = block._digest_hidden_rows(hidden)
        state_sha = pair.snapshot_digest(pair.capture_state(engine))
        many_report = many.report()
        fast_report = fast.report()
        state_batch_report = state_batch_core.report()
        gdn_rms_report = gdn_rms_many.report()
        attention_rms_report = attention_rms_many.report()
        head_many_report = head_many.report()

        if hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"current-best profile hidden anchor changed: {hidden_sha}")
        if state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"current-best profile state anchor changed: {state_sha}")
        if k3_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"expected one K3 stream, got {k3_bytes}")
        if many_report != EXPECTED_MANY:
            raise RuntimeError(f"unexpected quant-many coverage: {many_report}")
        if int(fast_report["calls"]) != EXPECTED_FAST_QUANT_CALLS:
            raise RuntimeError(f"unexpected fast-quantize calls: {fast_report}")
        if int(fast_report["values"]) != EXPECTED_FAST_QUANT_VALUES:
            raise RuntimeError(f"unexpected fast-quantize values: {fast_report}")
        if repeat_report != EXPECTED_REPEAT:
            raise RuntimeError(f"unexpected repeat coverage: {repeat_report}")
        if state_batch_report != EXPECTED_BATCH:
            raise RuntimeError(f"unexpected state-batch coverage: {state_batch_report}")
        if gdn_rms_report != EXPECTED_GDN_RMS_MANY:
            raise RuntimeError(f"unexpected GDN RMSNorm-many coverage: {gdn_rms_report}")
        if attention_rms_report != EXPECTED_ATTN_RMS_MANY:
            raise RuntimeError(
                f"unexpected attention RMSNorm-many coverage: {attention_rms_report}")
        if head_many_report != EXPECTED_HEAD_MANY:
            raise RuntimeError(f"unexpected head RMSNorm-many coverage: {head_many_report}")
        if rms_wrapper != EXPECTED_ZERO_WRAPPER:
            raise RuntimeError(f"unexpected remaining scalar RMSNorm coverage: {rms_wrapper}")

        total_rms_rows = head_probe._combined_rms_rows(
            rms_wrapper, gdn_rms_report, attention_rms_report, head_many_report)
        total_rms_values = head_probe._combined_rms_values(
            rms_wrapper, gdn_rms_report, attention_rms_report, head_many_report)
        if total_rms_rows != EXPECTED_TOTAL_RMS_ROWS:
            raise RuntimeError(f"unexpected total RMSNorm rows: {total_rms_rows}")
        if total_rms_values != EXPECTED_TOTAL_RMS_VALUES:
            raise RuntimeError(f"unexpected total RMSNorm values: {total_rms_values}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("current-best profile requires direct I/O")
        for key, expected in EXPECTED_READER.items():
            if int(reader.get(key, -1)) != int(expected):
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")

        profile = timing.report()
        matvec_seconds = sum(timing.role_seconds.values())
        external_prepare_seconds = sum(timing.external_prepare_seconds.values())
        native_helper_seconds = sum(float(v) for v in native_stats.values())
        accounted_seconds = matvec_seconds + external_prepare_seconds + native_helper_seconds
        unprofiled_seconds = float(prefill_seconds) - accounted_seconds
        if unprofiled_seconds < -0.10:
            raise RuntimeError(
                f"current-best profile accounting became negative: {unprofiled_seconds}")

        payload = {
            "schema": "qwen38-current-best-profile-v1",
            "status": "PASS",
            "claim": "clean exact hotspot profile on the fully batched current-best 11-token decoder-prefill path",
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
            "fast_quantize": fast_report,
            "native_f32": native_f32.report(),
            "quant_many": many_report,
            "gdn_state_batch": state_batch_report,
            "gdn_rmsnorm_many": gdn_rms_report,
            "attention_rmsnorm_many": attention_rms_report,
            "attention_head_rmsnorm_many": head_many_report,
            "rmsnorm_remaining": rms_wrapper,
            "rmsnorm_total_rows": total_rms_rows,
            "rmsnorm_total_values": total_rms_values,
            "residual": residual_report,
            "attention_gate": attn_gate_report,
            "gdn_repeat_scale": repeat_report,
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "accounting_note": (
                "matvec_seconds includes activation preparation performed inside matvec_many; "
                "external_prepare_seconds counts prepare_many only when called outside matvec_many; "
                "fast_quantize_seconds is diagnostic and is not added again to accounted time"
            ),
            "profile_changes_arithmetic": False,
            "baseline_head_rmsnorm_many_ab_run": 33848401400,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_CURRENT_BEST_PROFILE_BITWISE_PASS")
        return payload
    finally:
        if fast is not None:
            fast.restore()
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    for name in (
        "model", "quant-lib", "many-lib", "state-lib", "batch-state-lib",
        "f32-lib", "attn-lib", "conv-lib", "gate-lib", "swiglu-lib",
        "rmsnorm-lib", "rmsnorm-many-lib", "head-rmsnorm-many-lib",
        "residual-lib", "attention-gate-lib", "repeat-lib", "inventory",
        "work-dir", "output",
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
