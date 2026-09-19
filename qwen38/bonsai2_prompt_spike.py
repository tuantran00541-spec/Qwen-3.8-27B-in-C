#!/usr/bin/env python3
"""LAB-ONLY real prompt spike for Ternary Bonsai 2 27B.

This is intentionally not wired into the user-facing launcher yet. It reuses the
verified Bonsai 2 PTQ1 native-C/K3 path and the portable persistent GDN state pool
to exercise:
  text -> pinned Qwen3.8 tokenizer -> multi-token prefill -> stateful decode
  -> streamed low-bit LM head -> greedy text.

The spike is kept on test/bonsai2-27b until a real prompt run is inspected.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import bonsai2_full64_one_token as base
import bonsai2_two_token as t2
from bonsai2_quant_runtime import Bonsai2NativeRuntime
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk
import qwen35_full_attn_layer3_gate as attn
import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_ggml_exact as exact
import qwen35_k3_full64_ggml_rmsnorm as rmswrap
import qwen35_k3_generate as textgen

N_LAYER = 64
EOS_IDS = {248044, 248046}


class Bonsai2DecoderReader(K3Trunk):
    def __init__(
        self,
        bin_path: Path,
        index_path: Path,
        manifest: dict[str, Any],
        *,
        resident_decoder: bool,
        prefer_direct_io: bool = True,
    ) -> None:
        layers = list(manifest["layers"])
        if not layers:
            raise ValueError("decoder manifest has no layers")
        self.resident_decoder = bool(resident_decoder)
        self.decoder_resident_bytes = sum(int(x["read_bytes"]) for x in layers)
        max_layer_bytes = max(int(x["read_bytes"]) for x in layers)
        if self.resident_decoder:
            budget_bytes = self.decoder_resident_bytes
            max_pinned = len(layers)
        else:
            budget_bytes = 2 * max_layer_bytes
            max_pinned = 0
        super().__init__(
            bin_path,
            index_path,
            budget_bytes=budget_bytes,
            want_ring=2,
            max_pinned=max_pinned,
            prefer_direct_io=prefer_direct_io,
        )

    def report(self) -> dict[str, Any]:
        out = super().report()
        out["resident_decoder"] = self.resident_decoder
        out["decoder_resident_bytes"] = self.decoder_resident_bytes
        return out


def make_decoder_reader(
    bin_path: Path,
    index_path: Path,
    manifest: dict[str, Any],
    *,
    resident_decoder: bool,
    prefer_direct_io: bool = True,
) -> Bonsai2DecoderReader:
    return Bonsai2DecoderReader(
        bin_path,
        index_path,
        manifest,
        resident_decoder=resident_decoder,
        prefer_direct_io=prefer_direct_io,
    )


def f32(x: float) -> float:
    return exact.f32(x)


def addf(a: float, b: float) -> float:
    return t2.addf(a, b)


def mulf(a: float, b: float) -> float:
    return t2.mulf(a, b)


def profile_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    wall_seconds: float,
) -> dict[str, Any]:
    wall_seconds = max(0.0, float(wall_seconds))

    def timing_delta(
        seconds_key: str,
        calls_key: str,
        *,
        prefix: str = "",
    ) -> tuple[dict[str, float], dict[str, int]]:
        before_seconds = dict(before.get(seconds_key, {}))
        after_seconds = dict(after.get(seconds_key, {}))
        before_calls = dict(before.get(calls_key, {}))
        after_calls = dict(after.get(calls_key, {}))

        seconds: dict[str, float] = {}
        calls: dict[str, int] = {}
        for key in sorted(set(before_seconds) | set(after_seconds)):
            name = f"{prefix}{key}"
            seconds[name] = max(
                0.0,
                float(after_seconds.get(key, 0.0))
                - float(before_seconds.get(key, 0.0)),
            )
        for key in sorted(set(before_calls) | set(after_calls)):
            name = f"{prefix}{key}"
            calls[name] = max(
                0,
                int(after_calls.get(key, 0))
                - int(before_calls.get(key, 0)),
            )
        return seconds, calls

    components, component_calls = timing_delta(
        "lowbit_runtime_timing_seconds",
        "lowbit_runtime_timing_calls",
    )
    state_seconds, state_calls = timing_delta(
        "gdn_state_timing_seconds",
        "gdn_state_timing_calls",
        prefix="gdn_state_",
    )
    components.update(state_seconds)
    component_calls.update(state_calls)

    before_reader = dict(before.get("reader", {}))
    after_reader = dict(after.get("reader", {}))
    reader = {
        key: max(
            0,
            int(after_reader.get(key, 0)) - int(before_reader.get(key, 0)),
        )
        for key in ("bytes_read", "hits", "misses")
    }

    tracked_seconds = sum(components.values())
    untracked_seconds = max(0.0, wall_seconds - tracked_seconds)
    tracked_fraction = (
        tracked_seconds / wall_seconds
        if wall_seconds > 0.0
        else 0.0
    )
    return {
        "wall_seconds": wall_seconds,
        "components_seconds": components,
        "component_calls": component_calls,
        "reader": reader,
        "tracked_seconds": tracked_seconds,
        "untracked_seconds": untracked_seconds,
        "tracked_fraction": tracked_fraction,
    }


def softmax_many(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("softmax requires at least one score")
    m = max(float(x) for x in scores)
    exps = [f32(exact.expf(f32(float(x) - m))) for x in scores]
    denom = f32(0.0)
    for value in exps:
        denom = addf(denom, value)
    return [f32(value / denom) for value in exps]


def recurrent_step(
    runtime: Bonsai2NativeRuntime,
    state_lib: t2.GDNStateRuntime,
    state,
    history: Sequence[Sequence[float]],
    view,
    metas,
    vec,
    hidden: Sequence[float],
    layer: int,
) -> tuple[list[float], list[float]]:
    p = f"blk.{layer}"
    x = runtime.rms_norm(hidden, vec("attn_norm.weight"), eps=gdn.RMS_EPS)

    prepared = runtime.prepare_activation(f"{p}.attn_qkv.weight", x)
    qkv = runtime.matvec_prepared(
        view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], prepared
    )
    z = runtime.matvec_prepared(
        view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], prepared
    )
    beta_raw = runtime.matvec(
        view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], x
    )
    alpha = runtime.matvec(
        view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], x
    )
    beta = [exact.sigmoid_f32(value) for value in beta_raw]
    dt = vec("ssm_dt.bias")
    aa = vec("ssm_a")
    gate = [
        mulf(aa[h], t2.softplusf(addf(alpha[h], dt[h])))
        for h in range(gdn.V_HEADS)
    ]

    kernels = vec("ssm_conv1d.weight")
    conv = runtime.gdn_conv_silu(qkv, history, kernels)

    q = conv[: gdn.KEY_DIM]
    k = conv[gdn.KEY_DIM : 2 * gdn.KEY_DIM]
    v = conv[2 * gdn.KEY_DIM :]
    qn = gdn.flatten([
        gdn.l2_norm(head) for head in gdn.split_heads(q, gdn.K_HEADS)
    ])
    kn = gdn.flatten([
        gdn.l2_norm(head) for head in gdn.split_heads(k, gdn.K_HEADS)
    ])
    q48, k48 = runtime.gdn_repeat_scale(
        qn,
        kn,
        repeats=gdn.V_HEADS // gdn.K_HEADS,
        scale=t2.SCALE_GDN,
    )

    out_buf = (t2.ctypes.c_float * gdn.VALUE_DIM)()
    rc = state_lib.step(
        state,
        t2.carr(q48),
        t2.carr(k48),
        t2.carr(v),
        t2.carr(gate),
        t2.carr(beta),
        out_buf,
    )
    if rc != 0:
        raise RuntimeError(f"layer {layer}: GDN state pool rc={rc}")
    core = [float(out_buf[i]) for i in range(gdn.VALUE_DIM)]

    norm_w = vec("ssm_norm.weight")
    gated = runtime.gdn_norm_gate(core, norm_w, z, eps=gdn.RMS_EPS)

    linear = runtime.matvec(
        view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gated
    )
    residual = runtime.residual_add(hidden, linear)
    post = runtime.rms_norm(residual, vec("post_attention_norm.weight"), eps=gdn.RMS_EPS)
    ffn = t2.ffn(runtime, view, metas, p, post)
    return runtime.residual_add(residual, ffn), qkv


def full_attention_step(
    runtime: Bonsai2NativeRuntime,
    cache: dict[str, list[list[float]]],
    view,
    metas,
    vec,
    hidden: Sequence[float],
    layer: int,
    position: int,
) -> list[float]:
    p = f"blk.{layer}"
    x = runtime.rms_norm(hidden, vec("attn_norm.weight"), eps=gdn.RMS_EPS)

    prepared = runtime.prepare_activation(f"{p}.attn_q.weight", x)
    qg = runtime.matvec_prepared(
        view("attn_q.weight"), metas[f"{p}.attn_q.weight"], prepared
    )
    k = runtime.matvec_prepared(
        view("attn_k.weight"), metas[f"{p}.attn_k.weight"], prepared
    )
    v = runtime.matvec_prepared(
        view("attn_v.weight"), metas[f"{p}.attn_v.weight"], prepared
    )

    q, gate = attn.split_q_gate(qg)
    q = runtime.rms_norm(q, vec("attn_q_norm.weight"), rows=attn.N_HEAD, eps=attn.RMS_EPS)
    k = runtime.rms_norm(k, vec("attn_k_norm.weight"), rows=attn.N_HEAD_KV, eps=attn.RMS_EPS)
    q_rope = t2.rope_text_neox(q, attn.N_HEAD, position)
    k_rope = t2.rope_text_neox(k, attn.N_HEAD_KV, position)

    cache["k"].append(attn.f16_roundtrip(k_rope))
    cache["v"].append(attn.f16_roundtrip(v))
    pregate = runtime.attention_core(
        layer,
        q_rope,
        cache,
        q_heads=attn.N_HEAD,
        kv_heads=attn.N_HEAD_KV,
        head_dim=attn.HEAD_DIM,
        scale=t2.SCALE_ATTN,
    )

    gated = runtime.attention_sigmoid_mul(pregate, gate)
    attn_out = runtime.matvec(
        view("attn_output.weight"), metas[f"{p}.attn_output.weight"], gated
    )
    residual = runtime.residual_add(hidden, attn_out)
    post = runtime.rms_norm(residual, vec("post_attention_norm.weight"), eps=gdn.RMS_EPS)
    ffn = t2.ffn(runtime, view, metas, p, post)
    return runtime.residual_add(residual, ffn)


class StatefulBonsai2Generator:
    def __init__(
        self,
        model: Path,
        native_lib: Path,
        state_lib_path: Path,
        work_dir: Path,
        threads: int,
        resident_decoder: bool = False,
    ) -> None:
        exact.install()
        self.model = model
        self.directory = parse_gguf(model)
        if self.directory.metadata.get("general.architecture") != "qwen35":
            raise ValueError("Bonsai 2 prompt spike requires qwen35 GGUF")
        self.tensors = self.directory.by_name()
        self.runtime = Bonsai2NativeRuntime(
            native_lib,
            self.directory.metadata,
            threads=threads,
            max_rows=base.VOCAB,
        )
        self.state_lib = t2.load_state_lib(state_lib_path, threads)
        self.states = {
            layer: (t2.ctypes.c_float * t2.STATE_ELEMS)()
            for layer in range(N_LAYER)
            if layer % 4 != 3
        }
        self.conv_history: dict[int, list[array]] = {
            layer: []
            for layer in range(N_LAYER)
            if layer % 4 != 3
        }
        self.caches = {
            layer: {"k": [], "v": []}
            for layer in range(N_LAYER)
            if layer % 4 == 3
        }
        self.position = 0

        work_dir.mkdir(parents=True, exist_ok=True)
        trunk = work_dir / "decoder64.k3.bin"
        manifest_path = work_dir / "decoder64.k3.json"
        self.manifest = pack_gguf_layers(
            self.directory,
            trunk,
            manifest_path,
            layers=range(N_LAYER),
            model_id=base.MODEL_ID,
            revision=base.MODEL_REVISION,
            source_sha256=base.MODEL_SHA256,
            expected_layers=N_LAYER,
        )
        self.reader = make_decoder_reader(
            trunk,
            manifest_path,
            self.manifest,
            resident_decoder=resident_decoder,
            prefer_direct_io=True,
        )
        self.output_norm = base.read_f32_global(
            model, self.tensors["output_norm.weight"]
        )

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.state_lib.close()
            self.runtime.close()

    def step(self, token_id: int) -> list[float]:
        hidden = base.embedding_row(
            self.model, self.directory, self.runtime, int(token_id)
        )
        position = self.position

        for layer in range(N_LAYER):
            bound = self.reader.bind(layer)
            if layer + 1 < N_LAYER:
                self.reader.prefetch(layer + 1)
            metas = base.layer_meta(self.manifest, layer)
            prefix = f"blk.{layer}"

            def view(suffix: str):
                return self.reader.tensor_view(bound, f"{prefix}.{suffix}")

            def vec(suffix: str) -> list[float]:
                return gdn.f32_vector(view(suffix))

            if layer % 4 == 3:
                hidden = full_attention_step(
                    self.runtime,
                    self.caches[layer],
                    view,
                    metas,
                    vec,
                    hidden,
                    layer,
                    position,
                )
            else:
                hidden, qkv = recurrent_step(
                    self.runtime,
                    self.state_lib,
                    self.states[layer],
                    self.conv_history[layer],
                    view,
                    metas,
                    vec,
                    hidden,
                    layer,
                )
                history = self.conv_history[layer]
                history.append(array("f", qkv))
                if len(history) > 3:
                    del history[0]
            bound.release()

        self.position += 1
        return hidden

    def logits(self, hidden: Sequence[float]) -> list[float]:
        normalized = self.runtime.rms_norm(hidden, self.output_norm, eps=gdn.RMS_EPS)
        return base.stream_lowbit_logits(
            self.model,
            self.tensors["output.weight"],
            self.runtime,
            normalized,
        )

    def profile_snapshot(self) -> dict[str, Any]:
        runtime = self.runtime.report()
        state = self.state_lib.report()
        reader = self.reader.report()
        return {
            "lowbit_runtime_timing_seconds": dict(
                runtime.get("timing_seconds", {})
            ),
            "lowbit_runtime_timing_calls": dict(
                runtime.get("timing_calls", {})
            ),
            "gdn_state_timing_seconds": dict(
                state.get("timing_seconds", {})
            ),
            "gdn_state_timing_calls": dict(
                state.get("timing_calls", {})
            ),
            "reader": {
                key: int(reader.get(key, 0))
                for key in ("bytes_read", "hits", "misses")
            },
        }

    def report(self) -> dict[str, Any]:
        conv_bytes = sum(
            len(history) * gdn.CONV_DIM * 4
            for history in self.conv_history.values()
        )
        kv_bytes = 0
        for cache in self.caches.values():
            kv_bytes += sum(len(value) * 2 for value in cache["k"])
            kv_bytes += sum(len(value) * 2 for value in cache["v"])
        return {
            "position": self.position,
            "gdn_state_bytes_f32": 48 * t2.STATE_BYTES_PER_LAYER,
            "conv_history_bytes_f32": conv_bytes,
            "attention_kv_bytes_f16": kv_bytes,
            "reader": self.reader.report(),
            "lowbit_runtime": self.runtime.report(),
            "gdn_state_runtime": self.state_lib.report(),
        }


def run(
    model: Path,
    native_lib: Path,
    state_lib: Path,
    tokenizer_json: Path,
    prompt: str,
    max_new_tokens: int,
    work_dir: Path,
    output: Path,
    threads: int,
    resident_decoder: bool = False,
    expected_text: str | None = None,
    stream_text: bool = False,
    json_events: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    tokenizer = textgen.load_tokenizer(tokenizer_json)
    rendered, prompt_ids = textgen.encode_prompt(tokenizer, prompt, raw=False)
    if not prompt_ids:
        raise RuntimeError("empty prompt tokenization")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    engine = StatefulBonsai2Generator(
        model,
        native_lib,
        state_lib,
        work_dir,
        threads,
        resident_decoder=resident_decoder,
    )
    generated: list[int] = []
    token_reports: list[dict[str, Any]] = []
    prefill_seconds: list[float] = []
    decode_seconds: list[float] = []

    try:
        hidden = None
        for index, token_id in enumerate(prompt_ids):
            t0 = time.monotonic()
            hidden = engine.step(token_id)
            elapsed = time.monotonic() - t0
            prefill_seconds.append(elapsed)
            if json_events:
                print(json.dumps({
                    "phase": "prefill",
                    "index": index,
                    "token": int(token_id),
                    "position": engine.position,
                    "seconds": elapsed,
                }), flush=True)
        assert hidden is not None

        generation_started = time.monotonic()
        generation_profile_before = engine.profile_snapshot()
        first_token_seconds: float | None = None

        for index in range(max_new_tokens):
            cycle_started = time.monotonic()
            profile_before = engine.profile_snapshot()
            t0 = time.monotonic()
            logits = engine.logits(hidden)
            top5 = base.topk(logits, 5)
            token_id = int(top5[0]["token"])
            generated.append(token_id)
            piece = tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
            logit_seconds = time.monotonic() - t0
            report = {
                "index": index,
                "position": engine.position,
                "token": token_id,
                "piece": piece,
                "top5": top5,
                "logit_seconds": logit_seconds,
            }
            if first_token_seconds is None:
                first_token_seconds = time.monotonic() - generation_started
            if stream_text:
                print(piece, end="", flush=True)

            eos = token_id in EOS_IDS
            includes_next_decoder_step = False
            if not eos and index + 1 < max_new_tokens:
                t1 = time.monotonic()
                hidden = engine.step(token_id)
                decode_seconds.append(time.monotonic() - t1)
                includes_next_decoder_step = True

            profile_after = engine.profile_snapshot()
            report["critical_path"] = profile_delta(
                profile_before,
                profile_after,
                wall_seconds=time.monotonic() - cycle_started,
            )
            report["critical_path"]["includes_next_decoder_step"] = (
                includes_next_decoder_step
            )
            token_reports.append(report)

            if json_events:
                print(json.dumps({
                    "phase": "decode",
                    **report,
                }, ensure_ascii=False), flush=True)

            if eos:
                break

        generated_text = tokenizer.decode(
            generated, skip_special_tokens=False
        )
        if stream_text:
            print(flush=True)

        stop_reason = (
            "eos"
            if generated and generated[-1] in EOS_IDS
            else "max_new_tokens"
        )
        generation_seconds = time.monotonic() - generation_started
        decode_critical_path = profile_delta(
            generation_profile_before,
            engine.profile_snapshot(),
            wall_seconds=generation_seconds,
        )
        exact_text_match = (
            None
            if expected_text is None
            else generated_text.strip() == expected_text.strip()
        )
        result = {
            "schema": "qwen38-bonsai2-generation-v2",
            "status": "PASS" if generated else "FAIL",
            "lab_only": True,
            "model_sha256": base.MODEL_SHA256,
            "prompt": prompt,
            "rendered_prompt": rendered,
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "generated_text": generated_text,
            "expected_text": expected_text,
            "exact_text_match": exact_text_match,
            "token_reports": token_reports,
            "decode_critical_path": decode_critical_path,
            "stop_reason": stop_reason,
            "completion_truncated": stop_reason == "max_new_tokens",
            "threads": threads,
            "resident_decoder": resident_decoder,
            "timing": {
                "prefill_total_seconds": sum(prefill_seconds),
                "prefill_mean_seconds_per_token": (
                    sum(prefill_seconds) / len(prefill_seconds)
                ),
                "decode_step_total_seconds_excluding_logits": sum(decode_seconds),
                "generation_seconds_including_logits": generation_seconds,
                "time_to_first_token_seconds": first_token_seconds,
                "tokens_per_second": (
                    len(generated) / generation_seconds
                    if generation_seconds > 0.0
                    else 0.0
                ),
                "elapsed_seconds": time.monotonic() - started,
            },
            "state": engine.report(),
            "max_rss_gib": t2.rss_gib(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": result["status"],
            "prompt_token_count": result["prompt_token_count"],
            "generated_token_ids": result["generated_token_ids"],
            "generated_text": generated_text,
            "expected_text": expected_text,
            "exact_text_match": exact_text_match,
            "stop_reason": result["stop_reason"],
            "decode_critical_path": result["decode_critical_path"],
            "timing": result["timing"],
            "state": result["state"],
            "max_rss_gib": result["max_rss_gib"],
        }, indent=2, ensure_ascii=False))
        if not generated:
            raise SystemExit(1)
        print("QWEN38_BONSAI2_REAL_PROMPT_SPIKE_PASS")
        return result
    finally:
        engine.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--native-lib", type=Path, required=True)
    ap.add_argument("--state-lib", type=Path, required=True)
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resident-decoder", action="store_true")
    ap.add_argument("--expected-text")
    ap.add_argument("--stream-text", action="store_true")
    ap.add_argument("--no-json-events", action="store_true")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run(
        args.model,
        args.native_lib,
        args.state_lib,
        args.tokenizer_json,
        args.prompt,
        args.max_new_tokens,
        args.work_dir,
        args.output,
        args.threads,
        args.resident_decoder,
        args.expected_text,
        args.stream_text,
        not args.no_json_events,
    )


if __name__ == "__main__":
    main()
