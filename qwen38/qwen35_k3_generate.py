#!/usr/bin/env python3
"""Stateful N-token Qwen3.8-27B K3 text generator.

This extends the proven two-token runtime without changing that evidence path:
- 48 persistent F32 Gated-DeltaNet states through the native AR kernel;
- 3-token recurrent convolution history (kernel size 4);
- growing default-F16 K/V cache for the 16 full-attention layers;
- text MRoPE positions;
- streamed one-slot K3 weights and streamed Q8_0 LM head;
- official Qwen3.8 tokenizer.json for encode/decode;
- exact plain-user no-thinking chat envelope from the pinned official template.

The first implementation is deliberately greedy-only. Sampling can be layered on
once short multi-token behavioral gates are stable.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as base
import qwen35_k3_full64_ggml_exact as exact
import qwen35_k3_full64_ggml_rmsnorm as rmswrap
import qwen35_k3_two_token as t2
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

N_LAYER = 64
TOKENIZER_REVISION = gdn.REVISION
EOS_IDS = {248044, 248046}  # model EOS / official chat-template EOS
DEFAULT_MAX_PROMPT_TOKENS = 64


def f32(x: float) -> float:
    return exact.f32(x)


def addf(a: float, b: float) -> float:
    return t2.addf(a, b)


def mulf(a: float, b: float) -> float:
    return t2.mulf(a, b)


def build_chat_prompt(user_text: str) -> str:
    """Official plain-text chat-template specialization with thinking disabled."""
    return (
        "<|im_start|>user\n"
        + user_text.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "<think>\n\n</think>\n\n"
    )


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - exercised by real workflow
        raise RuntimeError("tokenizers package is required for text generation") from exc
    return Tokenizer.from_file(str(path))


def encode_prompt(tokenizer, text: str, *, raw: bool) -> tuple[str, list[int]]:
    rendered = text if raw else build_chat_prompt(text)
    ids = list(tokenizer.encode(rendered, add_special_tokens=False).ids)
    if not ids:
        raise ValueError("prompt tokenized to an empty sequence")
    return rendered, ids


def softmax_many(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("softmax requires at least one score")
    m = max(float(x) for x in scores)
    exps = [f32(exact.expf(f32(float(x) - m))) for x in scores]
    denom = f32(0.0)
    for x in exps:
        denom = addf(denom, x)
    return [f32(x / denom) for x in exps]


def recurrent_step(
    runtime,
    state_lib,
    state,
    history: Sequence[Sequence[float]],
    view,
    metas,
    vec,
    hidden: Sequence[float],
    layer: int,
):
    p = f"blk.{layer}"
    x = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qkv = runtime.matvec(view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], x)
    z = runtime.matvec(view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], x)
    beta_raw = runtime.matvec(view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], x)
    alpha = runtime.matvec(view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], x)
    beta = [exact.sigmoid_f32(v) for v in beta_raw]
    dt = vec("ssm_dt.bias")
    aa = vec("ssm_a")
    gate = [mulf(aa[h], t2.softplusf(addf(alpha[h], dt[h]))) for h in range(gdn.V_HEADS)]

    kernels = vec("ssm_conv1d.weight")
    prior = list(history[-3:])
    conv = [0.0] * gdn.CONV_DIM
    for c in range(gdn.CONV_DIM):
        cur = mulf(qkv[c], kernels[c * gdn.CONV_KERNEL + 3])
        # history is oldest -> newest. The newest token multiplies kernel[2],
        # then kernel[1], then kernel[0], exactly matching the 4-wide causal conv.
        for lag, old in enumerate(reversed(prior), start=1):
            cur = addf(cur, mulf(old[c], kernels[c * gdn.CONV_KERNEL + 3 - lag]))
        conv[c] = t2.siluf(cur)

    q = conv[: gdn.KEY_DIM]
    k = conv[gdn.KEY_DIM : 2 * gdn.KEY_DIM]
    v = conv[2 * gdn.KEY_DIM :]
    qn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(q, gdn.K_HEADS)])
    kn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(k, gdn.K_HEADS)])
    q48 = [mulf(vv, t2.SCALE_GDN) for vv in t2.repeat_k_heads(qn)]
    k48 = t2.repeat_k_heads(kn)

    out_buf = (t2.ctypes.c_float * gdn.VALUE_DIM)()
    rc = state_lib.qwen_gdn_ar_step_f32(
        state,
        t2.carr(q48),
        t2.carr(k48),
        t2.carr(v),
        t2.carr(gate),
        t2.carr(beta),
        out_buf,
    )
    if rc != 0:
        raise RuntimeError(f"layer {layer}: GDN state kernel rc={rc}")
    core = [float(out_buf[i]) for i in range(gdn.VALUE_DIM)]

    norm_w = vec("ssm_norm.weight")
    core_h = gdn.split_heads(core, gdn.V_HEADS)
    z_h = gdn.split_heads(z, gdn.V_HEADS)
    gated: list[float] = []
    for ch, zh in zip(core_h, z_h):
        nh = rmswrap.ggml_rms_norm(ch, norm_w, gdn.RMS_EPS)
        gated.extend(mulf(nh[d], t2.siluf(zh[d])) for d in range(gdn.HEAD_DIM))
    linear = runtime.matvec(view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gated)
    residual = [addf(hidden[i], linear[i]) for i in range(gdn.HIDDEN)]
    post = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    fo = t2.ffn(runtime, view, metas, p, post)
    final = [addf(residual[i], fo[i]) for i in range(gdn.HIDDEN)]
    return final, qkv


def full_attn_step(runtime, cache, view, metas, vec, hidden: Sequence[float], layer: int, position: int):
    p = f"blk.{layer}"
    x = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qg = runtime.matvec(view("attn_q.weight"), metas[f"{p}.attn_q.weight"], x)
    q, gate = attn.split_q_gate(qg)
    q = attn.rms_norm_heads(q, attn.N_HEAD, vec("attn_q_norm.weight"))
    k = runtime.matvec(view("attn_k.weight"), metas[f"{p}.attn_k.weight"], x)
    v = runtime.matvec(view("attn_v.weight"), metas[f"{p}.attn_v.weight"], x)
    k = attn.rms_norm_heads(k, attn.N_HEAD_KV, vec("attn_k_norm.weight"))
    q_rope = t2.rope_text_neox(q, attn.N_HEAD, position)
    k_rope = t2.rope_text_neox(k, attn.N_HEAD_KV, position)
    cache["k"].append(attn.f16_roundtrip(k_rope))
    cache["v"].append(attn.f16_roundtrip(v))

    qh = attn.split_heads(q_rope, attn.N_HEAD)
    pregate: list[float] = []
    n_ctx = len(cache["k"])
    for qidx in range(attn.N_HEAD):
        kvh = qidx // attn.GQA_REPEAT
        qv = qh[qidx]
        scores: list[float] = []
        for ti in range(n_ctx):
            kh = cache["k"][ti][kvh * attn.HEAD_DIM : (kvh + 1) * attn.HEAD_DIM]
            score = f32(math.fsum(float(qv[d]) * float(kh[d]) for d in range(attn.HEAD_DIM)) * t2.SCALE_ATTN)
            scores.append(score)
        probs = softmax_many(scores)
        for d in range(attn.HEAD_DIM):
            acc = f32(0.0)
            for ti in range(n_ctx):
                vv = cache["v"][ti][kvh * attn.HEAD_DIM + d]
                acc = addf(acc, mulf(probs[ti], vv))
            pregate.append(acc)

    gs = [exact.sigmoid_f32(vv) for vv in gate]
    gated = [mulf(pregate[i], gs[i]) for i in range(attn.Q_DIM)]
    ao = runtime.matvec(view("attn_output.weight"), metas[f"{p}.attn_output.weight"], gated)
    residual = [addf(hidden[i], ao[i]) for i in range(gdn.HIDDEN)]
    post = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    fo = t2.ffn(runtime, view, metas, p, post)
    return [addf(residual[i], fo[i]) for i in range(gdn.HIDDEN)]


class StatefulK3Generator:
    def __init__(self, model: Path, native_lib: Path, state_lib_path: Path, inventory_json: Path, work_dir: Path):
        exact.install()
        inv = json.loads(inventory_json.read_text(encoding="utf-8"))
        if inv.get("status") != "PASS" or inv.get("sha256") != gdn.SHA256:
            raise RuntimeError("mixed decoder inventory is not PASS for the pinned GGUF")
        self.model = model
        self.directory = parse_gguf(model)
        self.tensors = self.directory.by_name()
        self.runtime = gdn.QuantRuntime(gdn._load_native(native_lib))
        self.state_lib = t2.load_state_lib(state_lib_path)
        self.states = {
            il: (t2.ctypes.c_float * t2.STATE_ELEMS)()
            for il in range(N_LAYER)
            if il % 4 != 3
        }
        self.conv_history: dict[int, list[array]] = {
            il: [] for il in range(N_LAYER) if il % 4 != 3
        }
        self.caches = {
            il: {"k": [], "v": []} for il in range(N_LAYER) if il % 4 == 3
        }
        self.position = 0

        trunk = work_dir / "decoder64.k3.bin"
        manifest_path = work_dir / "decoder64.k3.json"
        self.manifest = pack_gguf_layers(
            self.directory,
            trunk,
            manifest_path,
            layers=range(N_LAYER),
            model_id=gdn.MODEL_ID,
            revision=gdn.REVISION,
            source_sha256=gdn.SHA256,
            expected_layers=N_LAYER,
        )
        max_layer = max(int(x["read_bytes"]) for x in self.manifest["layers"])
        self.reader = K3Trunk(
            trunk,
            manifest_path,
            budget_bytes=2 * max_layer,
            want_ring=2,
            max_pinned=0,
            prefer_direct_io=True,
        )
        self.output_norm_w = base._read_f32_tensor(model, self.tensors["output_norm.weight"])

    def close(self) -> None:
        self.reader.close()

    def step(self, token_id: int) -> list[float]:
        hidden = gdn._embedding_row(self.model, self.directory, int(token_id))
        pos = self.position
        for il in range(N_LAYER):
            bound = self.reader.bind(il)
            if il + 1 < N_LAYER:
                self.reader.prefetch(il + 1)
            metas = base._layer_meta(self.manifest, il)
            p = f"blk.{il}"

            def view(suffix: str):
                return self.reader.tensor_view(bound, f"{p}.{suffix}")

            def vec(suffix: str):
                return gdn.f32_vector(view(suffix))

            if il % 4 == 3:
                hidden = full_attn_step(self.runtime, self.caches[il], view, metas, vec, hidden, il, pos)
            else:
                hidden, qkv = recurrent_step(
                    self.runtime,
                    self.state_lib,
                    self.states[il],
                    self.conv_history[il],
                    view,
                    metas,
                    vec,
                    hidden,
                    il,
                )
                hist = self.conv_history[il]
                hist.append(array("f", qkv))
                if len(hist) > 3:
                    del hist[0]
            bound.release()
        self.position += 1
        return hidden

    def logits(self, hidden: Sequence[float]) -> list[float]:
        result_norm = gdn.rms_norm(hidden, self.output_norm_w)
        return base._stream_q8_logits(self.model, self.tensors["output.weight"], self.runtime, result_norm)

    def state_report(self) -> dict[str, Any]:
        conv_f32 = sum(len(h) * gdn.CONV_DIM * 4 for h in self.conv_history.values())
        kv_f16 = 0
        for cache in self.caches.values():
            kv_f16 += sum(len(x) * 2 for x in cache["k"])
            kv_f16 += sum(len(x) * 2 for x in cache["v"])
        return {
            "position": self.position,
            "gdn_state_bytes_f32": 48 * t2.STATE_BYTES_PER_LAYER,
            "conv_history_bytes_f32": conv_f32,
            "attention_kv_bytes_f16": kv_f16,
            "reader": self.reader.report(),
        }


def generate(
    model: Path,
    native_lib: Path,
    state_lib: Path,
    inventory: Path,
    tokenizer_json: Path,
    prompt: str,
    raw_prompt: bool,
    max_new_tokens: int,
    max_prompt_tokens: int,
    work_dir: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    tokenizer = load_tokenizer(tokenizer_json)
    rendered, prompt_ids = encode_prompt(tokenizer, prompt, raw=raw_prompt)
    if len(prompt_ids) > max_prompt_tokens:
        raise RuntimeError(f"prompt has {len(prompt_ids)} tokens; limit is {max_prompt_tokens}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    engine = StatefulK3Generator(model, native_lib, state_lib, inventory, work_dir)
    generated: list[int] = []
    token_reports: list[dict[str, Any]] = []
    try:
        hidden = None
        for token_id in prompt_ids:
            hidden = engine.step(token_id)
        assert hidden is not None

        for new_index in range(max_new_tokens):
            logits = engine.logits(hidden)
            top5 = base._topk(logits, 5)
            token_id = int(top5[0]["token"])
            generated.append(token_id)
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            token_reports.append({
                "index": new_index,
                "position": engine.position,
                "token": token_id,
                "piece": piece,
                "top5": top5,
            })
            if token_id in EOS_IDS:
                break
            if new_index + 1 < max_new_tokens:
                hidden = engine.step(token_id)

        text = tokenizer.decode(generated, skip_special_tokens=False)
        result = {
            "schema": "qwen38-k3-generate-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "tokenizer_revision": TOKENIZER_REVISION,
            "mode": "raw" if raw_prompt else "chat_no_thinking",
            "prompt": prompt,
            "rendered_prompt": rendered,
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "generated_text": text,
            "token_reports": token_reports,
            "stop_reason": "eos" if generated and generated[-1] in EOS_IDS else "max_new_tokens",
            "state": engine.state_report(),
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": t2.rss_gib(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": result["status"],
            "prompt_token_count": result["prompt_token_count"],
            "generated_token_ids": generated,
            "generated_text": text,
            "stop_reason": result["stop_reason"],
            "elapsed_seconds": result["elapsed_seconds"],
            "max_rss_gib": result["max_rss_gib"],
        }, indent=2, ensure_ascii=False))
        return result
    finally:
        engine.close()


def sanity() -> None:
    exact.install()
    p = build_chat_prompt("Hi")
    assert p == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    probs = softmax_many([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 2e-6 and probs[2] > probs[1] > probs[0]
    history = [array("f", [float(i)]) for i in range(3)]
    assert [float(x[0]) for x in reversed(history[-3:])] == [2.0, 1.0, 0.0]
    assert 48 * t2.STATE_BYTES_PER_LAYER == 150994944
    print(json.dumps({
        "schema": "qwen38-k3-generate-sanity-v1",
        "status": "PASS",
        "tokenizer_revision": TOKENIZER_REVISION,
        "chat_mode": "official_plain_user_enable_thinking_false",
        "gdn_state_bytes_f32": 48 * t2.STATE_BYTES_PER_LAYER,
        "conv_history_tokens": 3,
        "eos_ids": sorted(EOS_IDS),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--state-lib", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--tokenizer-json", type=Path, required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--raw-prompt", action="store_true")
    run.add_argument("--max-new-tokens", type=int, default=4)
    run.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    generate(
        args.model,
        args.native_lib,
        args.state_lib,
        args.inventory,
        args.tokenizer_json,
        args.prompt,
        args.raw_prompt,
        args.max_new_tokens,
        args.max_prompt_tokens,
        args.work_dir,
        args.output,
    )


if __name__ == "__main__":
    main()
