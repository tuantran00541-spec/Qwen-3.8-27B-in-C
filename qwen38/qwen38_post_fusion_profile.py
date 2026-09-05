#!/usr/bin/env python3
"""Exact post-fusion profiler for the Qwen3.8 staged prefill path.

Profiles the current best evidence path after three independently proven exact
native changes:
  * full-attention causal score/softmax/value core;
  * GDN 4-tap causal convolution + SiLU;
  * GDN output per-head RMSNorm + SiLU gating.

The profiler does not change matrix arithmetic.  It replaces only the already
proven helpers above, then splits the remaining FFN into activation preparation,
gate/up/down quantized matvecs, and Python SwiGLU.  All final hidden vectors and
persistent decoder state must still match the established 11-token bitwise
anchors, and the decoder must consume exactly one K3 stream.
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
import qwen38_attention_core_exact_probe as attn_probe
import qwen38_gdn_output_gate_exact_probe as gate_probe
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_many_probe as prefill
import qwen38_k3_prompt_block_prefill_probe as block
from attention_core_runtime import ExactAttentionCore
from gdn_conv_silu_runtime import ExactGDNConvSilu
from gdn_output_gate_runtime import ExactGDNOutputGate
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many

PROMPT_IDS = list(prefill.PROMPT_IDS)
K3_STREAM_BYTES = prefill.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = prefill.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = prefill.KNOWN_STATE_SHA256


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def layer_from_prefix(prefix: str) -> int:
    parts = prefix.split(".")
    if len(parts) < 2 or parts[0] != "blk":
        raise ValueError(f"unexpected layer prefix: {prefix}")
    return int(parts[1])


def layer_kind(layer: int) -> str:
    return "attention" if (int(layer) + 1) % 4 == 0 else "gdn"


def tensor_role(name: str) -> str:
    for suffix in (
        "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
        "ssm_out.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ):
        if name.endswith(suffix):
            return suffix
    return name.rsplit(".", 1)[-1] if name else "unknown"


class PostFusionProfiler:
    def __init__(self) -> None:
        self.role_seconds: dict[str, float] = defaultdict(float)
        self.role_calls: dict[str, int] = defaultdict(int)
        self.prepare_seconds: dict[str, float] = defaultdict(float)
        self.prepare_calls: dict[str, int] = defaultdict(int)
        self.ffn_stage_seconds: dict[str, float] = defaultdict(float)
        self.ffn_stage_calls: dict[str, int] = defaultdict(int)
        self.ffn_kind_seconds: dict[str, float] = defaultdict(float)
        self.ffn_kind_calls: dict[str, int] = defaultdict(int)
        self.ffn_layers: dict[int, dict[str, Any]] = {}

    def record_matvec(self, meta: dict[str, Any], seconds: float) -> None:
        role = tensor_role(str(meta.get("name", "")))
        self.role_seconds[role] += float(seconds)
        self.role_calls[role] += 1

    def record_prepare(self, kind: str, seconds: float) -> None:
        self.prepare_seconds[str(kind)] += float(seconds)
        self.prepare_calls[str(kind)] += 1

    def record_ffn(
        self,
        prefix: str,
        *,
        total: float,
        prepare: float,
        gate: float,
        up: float,
        swiglu: float,
        down: float,
    ) -> None:
        layer = layer_from_prefix(prefix)
        kind = layer_kind(layer)
        values = {
            "total": float(total),
            "prepare_q6": float(prepare),
            "gate_matvec": float(gate),
            "up_matvec": float(up),
            "swiglu_python": float(swiglu),
            "down_matvec": float(down),
        }
        rec = {"layer": layer, "kind": kind, **values}
        self.ffn_layers[layer] = rec
        self.ffn_kind_seconds[kind] += float(total)
        self.ffn_kind_calls[kind] += 1
        for stage, seconds in values.items():
            self.ffn_stage_seconds[stage] += seconds
            self.ffn_stage_calls[stage] += 1

    def report(self) -> dict[str, Any]:
        stages = [
            {
                "stage": stage,
                "seconds": seconds,
                "calls": self.ffn_stage_calls[stage],
            }
            for stage, seconds in sorted(
                self.ffn_stage_seconds.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return {
            "ffn_stage_seconds": dict(self.ffn_stage_seconds),
            "ffn_stage_calls": dict(self.ffn_stage_calls),
            "ffn_stage_ranking": stages,
            "ffn_kind_seconds": dict(self.ffn_kind_seconds),
            "ffn_kind_calls": dict(self.ffn_kind_calls),
            "ffn_layers": [self.ffn_layers[i] for i in sorted(self.ffn_layers)],
            "matvec_role_seconds": dict(
                sorted(self.role_seconds.items(), key=lambda kv: kv[1], reverse=True)),
            "matvec_role_calls": dict(self.role_calls),
            "prepare_seconds_by_kind": dict(self.prepare_seconds),
            "prepare_calls_by_kind": dict(self.prepare_calls),
        }


def sanity() -> None:
    if layer_kind(0) != "gdn" or layer_kind(2) != "gdn" or layer_kind(3) != "attention":
        raise SystemExit("layer-kind sanity failed")
    if layer_kind(63) != "attention":
        raise SystemExit("last-layer kind sanity failed")
    if tensor_role("blk.8.ffn_gate.weight") != "ffn_gate.weight":
        raise SystemExit("tensor-role sanity failed")
    print("QWEN38_POST_FUSION_PROFILE_SANITY PASS")


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)
        profiler = PostFusionProfiler()

        attention_core = ExactAttentionCore(args.attn_lib)
        conv_core = ExactGDNConvSilu(args.conv_lib)
        output_gate_core = ExactGDNOutputGate(args.gate_lib)
        native_stats = {
            "native_attention_core_seconds": 0.0,
            "native_conv_silu_seconds": 0.0,
            "native_output_gate_seconds": 0.0,
        }

        original_many = many.matvec_many
        original_prepare = many.prepare_many
        original_ffn = prefill._ffn_many
        original_attention = prefill._full_attention_layer_many
        original_recurrent = prefill._recurrent_layer_many

        def timed_prepare_many(xs, kind):
            t0 = time.monotonic()
            out = original_prepare(xs, kind)
            profiler.record_prepare(str(kind), time.monotonic() - t0)
            return out

        def timed_matvec_many(weights, meta, xs, prepared=None):
            t0 = time.monotonic()
            out = original_many(weights, meta, xs, prepared=prepared)
            profiler.record_matvec(meta, time.monotonic() - t0)
            return out

        def profiled_ffn(runtime, view, metas, prefix, xs):
            t_total = time.monotonic()

            t0 = time.monotonic()
            prepared = runtime.prepare_many(xs, "Q6_K")
            prepare_seconds = time.monotonic() - t0

            t0 = time.monotonic()
            gate = runtime.matvec_many(
                view("ffn_gate.weight"), metas[f"{prefix}.ffn_gate.weight"], xs,
                prepared=prepared)
            gate_seconds = time.monotonic() - t0

            t0 = time.monotonic()
            up = runtime.matvec_many(
                view("ffn_up.weight"), metas[f"{prefix}.ffn_up.weight"], xs,
                prepared=prepared)
            up_seconds = time.monotonic() - t0

            t0 = time.monotonic()
            sw = [
                [
                    gen.mulf(gen.t2.siluf(gate[j][i]), up[j][i])
                    for i in range(gdn.INTERMEDIATE)
                ]
                for j in range(len(xs))
            ]
            swiglu_seconds = time.monotonic() - t0

            t0 = time.monotonic()
            out = runtime.matvec_many(
                view("ffn_down.weight"), metas[f"{prefix}.ffn_down.weight"], sw)
            down_seconds = time.monotonic() - t0

            profiler.record_ffn(
                prefix,
                total=time.monotonic() - t_total,
                prepare=prepare_seconds,
                gate=gate_seconds,
                up=up_seconds,
                swiglu=swiglu_seconds,
                down=down_seconds,
            )
            return out

        def native_attention(engine_, hidden, il, pos0, view, metas, vec):
            local = {"native_attention_core_seconds": native_stats["native_attention_core_seconds"]}
            out = attn_probe._full_attention_layer_native(
                engine_, hidden, il, pos0, view, metas, vec, attention_core, local)
            native_stats["native_attention_core_seconds"] = local["native_attention_core_seconds"]
            return out

        def native_recurrent(engine_, hidden, il, view, metas, vec):
            local = {
                "native_conv_silu_seconds": native_stats["native_conv_silu_seconds"],
                "native_output_gate_seconds": native_stats["native_output_gate_seconds"],
            }
            out = gate_probe._recurrent_layer_native_conv_gate(
                conv_core, output_gate_core, local, engine_, hidden, il, view, metas, vec)
            native_stats["native_conv_silu_seconds"] = local["native_conv_silu_seconds"]
            native_stats["native_output_gate_seconds"] = local["native_output_gate_seconds"]
            return out

        many.prepare_many = timed_prepare_many
        many.matvec_many = timed_matvec_many
        prefill._ffn_many = profiled_ffn
        prefill._full_attention_layer_many = native_attention
        prefill._recurrent_layer_many = native_recurrent
        try:
            before = int(engine.reader.report()["bytes_read"])
            t0 = time.monotonic()
            hidden_rows = prefill.step_block_many(engine, PROMPT_IDS)
            prefill_seconds = time.monotonic() - t0
            k3_bytes = int(engine.reader.report()["bytes_read"]) - before
        finally:
            many.prepare_many = original_prepare
            many.matvec_many = original_many
            prefill._ffn_many = original_ffn
            prefill._full_attention_layer_many = original_attention
            prefill._recurrent_layer_many = original_recurrent

        hidden_sha = block._digest_hidden_rows(hidden_rows)
        final_state = pair.capture_state(engine)
        state_sha = pair.snapshot_digest(final_state)
        if hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"post-fusion profile hidden anchor changed: {hidden_sha}")
        if state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"post-fusion profile state anchor changed: {state_sha}")
        if k3_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"post-fusion profile expected one K3 stream, got {k3_bytes}")
        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("post-fusion profile requires direct I/O")

        attn_report = attention_core.report()
        conv_report = conv_core.report()
        gate_report = output_gate_core.report()
        if attn_report["calls"] != 16 * len(PROMPT_IDS):
            raise RuntimeError(f"unexpected native attention coverage: {attn_report}")
        if conv_report["calls"] != 48 or conv_report["tokens"] != 48 * len(PROMPT_IDS):
            raise RuntimeError(f"unexpected native conv coverage: {conv_report}")
        if gate_report["calls"] != 48 * len(PROMPT_IDS):
            raise RuntimeError(f"unexpected native output-gate coverage: {gate_report}")

        profile = profiler.report()
        payload = {
            "schema": "qwen38-post-fusion-profile-v1",
            "status": "PASS",
            "claim": "exact post-fusion performance profile; hidden/state anchors verified bitwise",
            "model_sha256": gdn.SHA256,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_sha256": hidden_sha,
            "state_sha256": state_sha,
            "prefill_seconds": prefill_seconds,
            "k3_bytes": k3_bytes,
            "native_seconds": native_stats,
            "native_attention": attn_report,
            "native_conv_silu": conv_report,
            "native_output_gate": gate_report,
            "profile": profile,
            "native_f32": native_f32.report(),
            "quant_many": many.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "research_basis": [
                "Liger and oneDNN both treat SwiGLU as a fusion opportunity (SiLU followed by multiply).",
                "Deep Kernel Fusion for Transformers (ACL 2026) identifies SwiGLU MLP traffic/cache reuse as a major inference optimization target.",
                "llama.cpp CPU work distribution parallelizes independent output rows; this preserves per-row reduction order and is a later exact-threading candidate if matvec remains dominant.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_POST_FUSION_PROFILE_BITWISE_PASS")
        return payload
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--quant-lib", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--f32-lib", type=Path, required=True)
    r.add_argument("--attn-lib", type=Path, required=True)
    r.add_argument("--conv-lib", type=Path, required=True)
    r.add_argument("--gate-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
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
