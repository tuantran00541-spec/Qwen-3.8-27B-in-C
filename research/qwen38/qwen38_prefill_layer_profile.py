#!/usr/bin/env python3
"""Bounded per-layer profiler for the proven exact staged Qwen3.8 prefill path.

The profiler does not change arithmetic or scheduling. It wraps the existing
staged matvec-many helpers and runtime methods to attribute wall time to:
  * recurrent GDN vs full-attention layers;
  * quantized/F32 matrix work by tensor role;
  * activation preparation (a subset of matrix time);
  * the remaining causal / normalization / Python work inside each layer.

A prefix of the real compact math prompt keeps CI bounded while preserving the
same target runtime path. This is performance evidence, not a correctness gate.
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
import qwen38_k3_prompt_block_many_probe as prefill_many
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many

K3_STREAM_BYTES = 21_127_430_144
COMPACT_PROMPT = """4 final answers only.
1 sqrt(x+6)+sqrt(x-3)=5,x>=3
2 5R4B3G urn,3 draws no replacement:P(exactly 2 colors)
3 7^2026 mod1000
4 positive a<=b,1/a+1/b=1/6:all pairs"""


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def emit(event: str, **fields: Any) -> None:
    payload = {"marker": "QWEN38_PREFILL_LAYER_PROFILE", "event": event}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def tensor_role(name: str) -> str:
    for suffix in (
        "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
        "ssm_out.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ):
        if name.endswith(suffix):
            return suffix
    return name.rsplit(".", 1)[-1] if name else "unknown"


class TimingProfiler:
    def __init__(self) -> None:
        self.current_layer: int | None = None
        self.current_kind: str | None = None
        self.layers: dict[int, dict[str, Any]] = {}
        self.role_seconds: dict[str, float] = defaultdict(float)
        self.role_calls: dict[str, int] = defaultdict(int)
        self.prepare_seconds: dict[str, float] = defaultdict(float)
        self.prepare_calls: dict[str, int] = defaultdict(int)

    def begin_layer(self, layer: int, kind: str) -> None:
        self.current_layer = int(layer)
        self.current_kind = kind
        self.layers[int(layer)] = {
            "layer": int(layer),
            "kind": kind,
            "seconds": 0.0,
            "matvec_seconds_inclusive": 0.0,
            "matvec_calls": 0,
            "prepare_seconds_subset": 0.0,
            "prepare_calls": 0,
            "roles": {},
        }
        emit("layer_begin", layer=int(layer), kind=kind)

    def end_layer(self, layer: int, seconds: float) -> None:
        rec = self.layers[int(layer)]
        rec["seconds"] = float(seconds)
        rec["non_matvec_seconds"] = max(0.0, float(seconds) - float(rec["matvec_seconds_inclusive"]))
        emit(
            "layer_end",
            layer=int(layer),
            kind=rec["kind"],
            seconds=rec["seconds"],
            matvec_seconds_inclusive=rec["matvec_seconds_inclusive"],
            non_matvec_seconds=rec["non_matvec_seconds"],
            prepare_seconds_subset=rec["prepare_seconds_subset"],
        )
        self.current_layer = None
        self.current_kind = None

    def record_matvec(self, meta: dict[str, Any], seconds: float) -> None:
        role = tensor_role(str(meta.get("name", "")))
        self.role_seconds[role] += float(seconds)
        self.role_calls[role] += 1
        if self.current_layer is None:
            return
        rec = self.layers[self.current_layer]
        rec["matvec_seconds_inclusive"] += float(seconds)
        rec["matvec_calls"] += 1
        roles = rec["roles"]
        r = roles.setdefault(role, {"seconds": 0.0, "calls": 0})
        r["seconds"] += float(seconds)
        r["calls"] += 1

    def record_prepare(self, kind: str, seconds: float) -> None:
        self.prepare_seconds[str(kind)] += float(seconds)
        self.prepare_calls[str(kind)] += 1
        if self.current_layer is None:
            return
        rec = self.layers[self.current_layer]
        rec["prepare_seconds_subset"] += float(seconds)
        rec["prepare_calls"] += 1


def install_profile_hooks(runtime, profiler: TimingProfiler) -> None:
    orig_matvec_many = runtime.matvec_many
    orig_prepare_many = runtime.prepare_many
    orig_recurrent = prefill_many._recurrent_layer_many
    orig_attention = prefill_many._full_attention_layer_many

    def profiled_prepare_many(xs, kind):
        t0 = time.monotonic()
        out = orig_prepare_many(xs, kind)
        profiler.record_prepare(str(kind), time.monotonic() - t0)
        return out

    def profiled_matvec_many(weights, meta, xs, prepared=None):
        t0 = time.monotonic()
        out = orig_matvec_many(weights, meta, xs, prepared=prepared)
        profiler.record_matvec(meta, time.monotonic() - t0)
        return out

    def profiled_recurrent(engine, hidden, il, view, metas, vec):
        profiler.begin_layer(int(il), "gdn")
        t0 = time.monotonic()
        try:
            return orig_recurrent(engine, hidden, il, view, metas, vec)
        finally:
            profiler.end_layer(int(il), time.monotonic() - t0)

    def profiled_attention(engine, hidden, il, pos0, view, metas, vec):
        profiler.begin_layer(int(il), "attention")
        t0 = time.monotonic()
        try:
            return orig_attention(engine, hidden, il, pos0, view, metas, vec)
        finally:
            profiler.end_layer(int(il), time.monotonic() - t0)

    runtime.prepare_many = profiled_prepare_many
    runtime.matvec_many = profiled_matvec_many
    prefill_many._recurrent_layer_many = profiled_recurrent
    prefill_many._full_attention_layer_many = profiled_attention


def summarize(profiler: TimingProfiler) -> dict[str, Any]:
    rows = [profiler.layers[i] for i in sorted(profiler.layers)]

    def aggregate(kind: str) -> dict[str, Any]:
        selected = [r for r in rows if r["kind"] == kind]
        total = sum(float(r["seconds"]) for r in selected)
        matvec = sum(float(r["matvec_seconds_inclusive"]) for r in selected)
        prepare = sum(float(r["prepare_seconds_subset"]) for r in selected)
        return {
            "layer_count": len(selected),
            "seconds": total,
            "matvec_seconds_inclusive": matvec,
            "non_matvec_seconds": max(0.0, total - matvec),
            "prepare_seconds_subset": prepare,
        }

    slowest = sorted(rows, key=lambda r: float(r["seconds"]), reverse=True)[:12]
    return {
        "gdn": aggregate("gdn"),
        "attention": aggregate("attention"),
        "role_seconds": dict(sorted(profiler.role_seconds.items(), key=lambda kv: kv[1], reverse=True)),
        "role_calls": dict(profiler.role_calls),
        "prepare_seconds_by_kind": dict(profiler.prepare_seconds),
        "prepare_calls_by_kind": dict(profiler.prepare_calls),
        "slowest_layers": slowest,
        "layers": rows,
    }


def run(args) -> dict[str, Any]:
    tokenizer = gen.load_tokenizer(args.tokenizer_json)
    rendered, full_ids = gen.encode_prompt(tokenizer, COMPACT_PROMPT, raw=False)
    if not full_ids:
        raise RuntimeError("compact prompt tokenized to empty sequence")
    count = min(int(args.profile_tokens), len(full_ids))
    if count < 1:
        raise ValueError("profile-tokens must be >= 1")
    prompt_ids = full_ids[:count]
    emit("profile_begin", full_prompt_tokens=len(full_ids), profile_tokens=count)

    t0 = time.monotonic()
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    engine_init_seconds = time.monotonic() - t0
    try:
        native = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        profiler = TimingProfiler()
        install_profile_hooks(many, profiler)

        reader_before = int(engine.reader.report()["bytes_read"])
        t0 = time.monotonic()
        hidden_rows = prefill_many.step_block_many(engine, prompt_ids)
        prefill_seconds = time.monotonic() - t0
        reader_bytes = int(engine.reader.report()["bytes_read"]) - reader_before

        if len(hidden_rows) != count:
            raise RuntimeError("profile hidden-row count mismatch")
        if reader_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"profile expected one K3 stream, got {reader_bytes}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("profile requires direct I/O")

        breakdown = summarize(profiler)
        accounted = breakdown["gdn"]["seconds"] + breakdown["attention"]["seconds"]
        payload = {
            "schema": "qwen38-prefill-layer-profile-v1",
            "status": "COMPLETE",
            "claim": "performance profile only; existing exact staged-prefill arithmetic path unchanged",
            "model_sha256": gdn.SHA256,
            "rendered_prompt": rendered,
            "full_prompt_token_count": len(full_ids),
            "profile_token_count": count,
            "profile_token_ids": prompt_ids,
            "engine_init_seconds": engine_init_seconds,
            "prefill_seconds": prefill_seconds,
            "profiled_layer_seconds": accounted,
            "unattributed_step_block_seconds": max(0.0, prefill_seconds - accounted),
            "k3_bytes": reader_bytes,
            "native_f32": native.report(),
            "quant_many": many.report(),
            "breakdown": breakdown,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        emit(
            "profile_complete",
            profile_tokens=count,
            prefill_seconds=prefill_seconds,
            gdn_seconds=breakdown["gdn"]["seconds"],
            gdn_non_matvec_seconds=breakdown["gdn"]["non_matvec_seconds"],
            attention_seconds=breakdown["attention"]["seconds"],
            attention_non_matvec_seconds=breakdown["attention"]["non_matvec_seconds"],
            k3_bytes=reader_bytes,
            max_rss_gib=payload["max_rss_gib"],
        )
        print("QWEN38_PREFILL_LAYER_PROFILE_COMPLETE")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert tensor_role("blk.3.attn_q.weight") == "attn_q.weight"
    assert tensor_role("blk.4.ssm_alpha.weight") == "ssm_alpha.weight"
    assert tensor_role("blk.7.ffn_down.weight") == "ffn_down.weight"
    print("QWEN38_PREFILL_LAYER_PROFILE_SANITY PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--quant-lib", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--tokenizer-json", type=Path, required=True)
    r.add_argument("--profile-tokens", type=int, default=44)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
