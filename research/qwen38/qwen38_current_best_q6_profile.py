#!/usr/bin/env python3
"""Clean exact profile of the promoted Qwen3.8 current-best quant stack.

Current-best experimental Linux quant path:
  * exact Q6_K persistent static row pool, pinned to two workers;
  * exact Q8_0 noalloc many-vector bridge;
  * fast activation input and F32 output marshaling.

All previously proven exact decoder helpers remain enabled.  The profiler is
measurement-only and additionally times main-thread K3 ``bind()`` wall time so
storage stalls exposed by faster Q6 compute are separated from residual Python
orchestration.  It does not change reader policy, ring residency, arithmetic,
or output-row reduction order.
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
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_prefill_probe as block
import qwen38_q6_persistent_pool_exact_probe as q6
from native_f32_runtime import enable_native_f32
from qwen38_current_best_runtime import (
    CURRENT_BEST_Q6_WORKERS,
    Qwen38CurrentBestQuantStack,
)

PROMPT_IDS = list(q6.PROMPT_IDS)
K3_STREAM_BYTES = q6.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = q6.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = q6.KNOWN_STATE_SHA256
EXPECTED_MANY = dict(q6.EXPECTED_MANY)
EXPECTED_Q6 = dict(q6.EXPECTED_Q6)
EXPECTED_READER = dict(q6.EXPECTED_READER)


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


def _install_bind_timer(reader):
    original = reader.bind
    stats: dict[str, Any] = {
        "calls": 0,
        "seconds": 0.0,
        "pending_requested_calls": 0,
        "pending_requested_seconds": 0.0,
        "layers": [],
    }

    def timed_bind(layer: int):
        requested = int(layer)
        pending = getattr(reader, "_pending", None)
        pending_requested = bool(pending is not None and int(pending[0]) == requested)
        t0 = time.monotonic()
        out = original(requested)
        dt = time.monotonic() - t0
        stats["calls"] += 1
        stats["seconds"] += dt
        if pending_requested:
            stats["pending_requested_calls"] += 1
            stats["pending_requested_seconds"] += dt
        stats["layers"].append({
            "layer": requested,
            "seconds": dt,
            "pending_requested": pending_requested,
        })
        return out

    reader.bind = timed_bind
    return original, stats


def _bind_report(stats: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        stats["layers"], key=lambda item: float(item["seconds"]), reverse=True)
    return {
        "calls": int(stats["calls"]),
        "seconds": float(stats["seconds"]),
        "pending_requested_calls": int(stats["pending_requested_calls"]),
        "pending_requested_seconds": float(stats["pending_requested_seconds"]),
        "top_layer_stalls": ranked[:12],
    }


def sanity() -> None:
    from qwen38_current_best_runtime import sanity as runtime_sanity

    runtime_sanity()
    q6.sanity()
    if len(PROMPT_IDS) != 11:
        raise SystemExit(f"unexpected prompt count: {len(PROMPT_IDS)}")
    if CURRENT_BEST_Q6_WORKERS != 2:
        raise SystemExit("current-best profile requires exactly two Q6 workers")
    if EXPECTED_Q6 != {"calls": 280, "vectors": 3080, "rows": 33882112}:
        raise SystemExit(f"unexpected Q6 contract: {EXPECTED_Q6}")
    print("QWEN38_CURRENT_BEST_Q6_PROFILE_SANITY PASS")


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    stack: Qwen38CurrentBestQuantStack | None = None
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        stack = Qwen38CurrentBestQuantStack(
            engine,
            args.pool_lib,
            q6_workers=CURRENT_BEST_Q6_WORKERS,
            max_vec=len(PROMPT_IDS),
        )
        runtime = stack.runtime
        if runtime is None:
            raise RuntimeError("current-best quant runtime did not initialize")

        timing = TimingProfile()
        original_many = runtime.matvec_many
        original_prepare = runtime.prepare_many

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
                dt = time.monotonic() - t0
                timing._matvec_depth -= 1
                timing.record_matvec(meta, dt)

        runtime.prepare_many = timed_prepare
        runtime.matvec_many = timed_many
        original_bind, bind_stats = _install_bind_timer(engine.reader)
        try:
            rec = q6._run_stack(engine, args, runtime)
        finally:
            engine.reader.bind = original_bind
            runtime.prepare_many = original_prepare
            runtime.matvec_many = original_many

        hidden_sha = block._digest_hidden_rows(rec["hidden"])
        state_sha = pair.snapshot_digest(pair.capture_state(engine))
        fast_report = stack.fast_quant.report()
        q6._check_common("current-best-q6", rec, fast_report)
        if hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"current-best hidden anchor changed: {hidden_sha}")
        if state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"current-best state anchor changed: {state_sha}")

        pool_report = runtime.pool_report()
        if int(pool_report["threads"]) != CURRENT_BEST_Q6_WORKERS:
            raise RuntimeError(f"unexpected Q6 worker report: {pool_report}")
        for key in ("calls", "native_calls", "vectors", "rows"):
            expected = EXPECTED_Q6["calls" if key in ("calls", "native_calls") else key]
            if int(pool_report[key]) != int(expected):
                raise RuntimeError(
                    f"unexpected Q6 pool {key}={pool_report[key]} expected={expected}")

        bind = _bind_report(bind_stats)
        if bind["calls"] != 64:
            raise RuntimeError(f"expected 64 decoder-layer binds, got {bind['calls']}")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("current-best Q6 profile requires direct I/O")
        for key, expected in EXPECTED_READER.items():
            if int(reader.get(key, -1)) != int(expected):
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")
        if int(reader.get("bytes_read", -1)) != K3_STREAM_BYTES:
            raise RuntimeError(
                f"reader bytes_read={reader.get('bytes_read')} expected={K3_STREAM_BYTES}")

        profile = timing.report()
        matvec_seconds = sum(float(v) for v in timing.role_seconds.values())
        external_prepare_seconds = sum(
            float(v) for v in timing.external_prepare_seconds.values())
        native_helper_seconds = sum(float(v) for v in rec["native_stats"].values())
        accounted_seconds = (
            matvec_seconds + external_prepare_seconds
            + native_helper_seconds + float(bind["seconds"])
        )
        residual_seconds = float(rec["seconds"]) - accounted_seconds
        if residual_seconds < -0.10:
            raise RuntimeError(
                f"current-best Q6 profile accounting became negative: {residual_seconds}")

        quant_stack_report = stack.report()
        payload = {
            "schema": "qwen38-current-best-q6-profile-v2",
            "status": "PASS",
            "claim": "clean exact hotspot and K3-bind profile on promoted Linux experimental Q6-pool2 current-best 11-token decoder-prefill path",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_sha256": hidden_sha,
            "state_sha256": state_sha,
            "prefill_seconds": rec["seconds"],
            "k3_bytes": rec["k3_bytes"],
            "matvec_seconds_inclusive": matvec_seconds,
            "q6_boundary_seconds": rec["q6_timing"]["seconds"],
            "reader_bind_seconds": bind["seconds"],
            "reader_pending_bind_seconds": bind["pending_requested_seconds"],
            "external_prepare_seconds": external_prepare_seconds,
            "native_helper_seconds": native_helper_seconds,
            "residual_non_bind_orchestration_seconds": residual_seconds,
            "profile": profile,
            "bind_profile": bind,
            "native_seconds": rec["native_stats"],
            "fast_quantize": fast_report,
            "native_f32": native_f32.report(),
            "quant_many": rec["many"],
            "q6_pool": pool_report,
            "gdn_state_batch": rec["state_batch"],
            "gdn_rmsnorm_many": rec["gdn_rms_many"],
            "attention_rmsnorm_many": rec["attention_rms_many"],
            "attention_head_rmsnorm_many": rec["head_many"],
            "rmsnorm_remaining": rec["rms_wrapper"],
            "residual": rec["residual"],
            "attention_gate": rec["attn_gate"],
            "gdn_repeat_scale": rec["repeat"],
            "quant_stack": quant_stack_report,
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "accounting_note": (
                "matvec_seconds includes Q6/Q8/F32 matvec-many boundary time; "
                "reader_bind_seconds is main-thread bind wall time and isolates exposed K3 prefetch stalls; "
                "fast_quantize_seconds and native Q6-pool seconds are diagnostic subcomponents and are not added again"
            ),
            "optimization": {
                "promoted_current_best": True,
                "linux_experimental": True,
                "q6_workers": CURRENT_BEST_Q6_WORKERS,
                "q6_static_disjoint_rows": True,
                "q8_noalloc": True,
                "reader_policy_change": False,
                "ring_change": False,
                "arithmetic_change": False,
                "windows_backend_promoted": False,
            },
            "proof_q6_pool_ab_run": 33851935544,
            "proof_q6_repeat_abba_run": 33853180708,
            "proof_q6_io_overlap_run": 33854341197,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_CURRENT_BEST_Q6_PROFILE_REAL_BITWISE_PASS")
        return payload
    finally:
        if stack is not None:
            stack.close()
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    for name in (
        "model", "quant-lib", "pool-lib", "state-lib", "batch-state-lib",
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
