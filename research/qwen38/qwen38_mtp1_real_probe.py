#!/usr/bin/env python3
"""Real Qwen3.8 MTP-1 acceptance probe against the exact target runtime.

This is deliberately not the final speculative verifier.  It establishes the
smallest useful semantic fact first: can the native blk.64 MTP head, fed with
the exact target h_nextn convention, draft the token that the target itself
produces next?

For the first context token, llama.cpp's MTP process uses a zero h input.  Each
later MTP position receives the target's normalized h_nextn from the previous
position.  We mirror that convention.  Prompt catch-up computes only the K/V
state required by the MTP attention block; the actual draft position executes
the complete MTP block and shared LM head.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import qwen35_full_attn_layer3_gate as attn
import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_full64_ggml_exact as exact
import qwen35_k3_generate as gen
import qwen35_k3_two_token as t2
import qwen38_mtp_quant_runtime as mq
from gguf_stream import parse_gguf

MTP_LAYER = 64
PREFIX = "blk.64."


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


class MTPBlock:
    def __init__(self, model: Path, base_native: Path, q4_native: Path):
        exact.install()
        self.model = model
        self.directory = parse_gguf(model)
        self.tensors = self.directory.by_name()
        self.runtime = mq.MTPQuantRuntime(
            gdn._load_native(base_native),
            mq.load_q4_native(q4_native),
        )
        self.cache = {"k": [], "v": []}
        self.buffers: dict[str, bytearray] = {}
        self.metas: dict[str, dict[str, Any]] = {}
        self._f32: dict[str, list[float]] = {}

        spans = [t for t in self.directory.tensors if t.name.startswith(PREFIX)]
        with model.open("rb", buffering=0) as f:
            for t in spans:
                f.seek(int(t.data_offset))
                raw = f.read(int(t.nbytes))
                if len(raw) != int(t.nbytes):
                    raise RuntimeError(f"short read for {t.name}")
                self.buffers[t.name] = bytearray(raw)
                self.metas[t.name] = {
                    "name": t.name,
                    "type_name": t.type_name,
                    "shape": list(t.shape),
                    "nbytes": int(t.nbytes),
                }
        if "blk.64.nextn.shared_head_norm.weight" not in self.buffers:
            raise RuntimeError("pinned GGUF unexpectedly lacks private MTP shared-head norm")
        self.resident_weight_bytes = sum(len(x) for x in self.buffers.values())

    def view(self, suffix: str) -> memoryview:
        return memoryview(self.buffers[PREFIX + suffix])

    def meta(self, suffix: str) -> dict[str, Any]:
        return self.metas[PREFIX + suffix]

    def vec(self, suffix: str) -> list[float]:
        if suffix not in self._f32:
            self._f32[suffix] = gdn.f32_vector(self.view(suffix))
        return self._f32[suffix]

    def _prepare(self, token_id: int, h_prev: Sequence[float]) -> list[float]:
        if len(h_prev) != gdn.HIDDEN:
            raise ValueError("MTP h_prev has wrong width")
        tok = gdn._embedding_row(self.model, self.directory, int(token_id))
        e = gdn.rms_norm(tok, self.vec("nextn.enorm.weight"))
        h = gdn.rms_norm(h_prev, self.vec("nextn.hnorm.weight"))
        concat = e + h
        return self.runtime.matvec(self.view("nextn.eh_proj.weight"), self.meta("nextn.eh_proj.weight"), concat)

    def _kv(self, cur: Sequence[float], position: int) -> tuple[list[float], list[float]]:
        x = gdn.rms_norm(cur, self.vec("attn_norm.weight"))
        k = self.runtime.matvec(self.view("attn_k.weight"), self.meta("attn_k.weight"), x)
        v = self.runtime.matvec(self.view("attn_v.weight"), self.meta("attn_v.weight"), x)
        k = attn.rms_norm_heads(k, attn.N_HEAD_KV, self.vec("attn_k_norm.weight"))
        k = t2.rope_text_neox(k, attn.N_HEAD_KV, position)
        return attn.f16_roundtrip(k), attn.f16_roundtrip(v)

    def catchup(self, token_id: int, h_prev: Sequence[float], position: int) -> None:
        cur = self._prepare(token_id, h_prev)
        k, v = self._kv(cur, position)
        self.cache["k"].append(k)
        self.cache["v"].append(v)

    def _ffn(self, x: Sequence[float]) -> list[float]:
        prepared = self.runtime.quantize(x, "Q4_0")
        gate = self.runtime.matvec(self.view("ffn_gate.weight"), self.meta("ffn_gate.weight"), x, prepared)
        up = self.runtime.matvec(self.view("ffn_up.weight"), self.meta("ffn_up.weight"), x, prepared)
        sw = [t2.mulf(t2.siluf(gate[i]), up[i]) for i in range(gdn.INTERMEDIATE)]
        return self.runtime.matvec(self.view("ffn_down.weight"), self.meta("ffn_down.weight"), sw)

    def draft(self, token_id: int, h_prev: Sequence[float], position: int) -> dict[str, Any]:
        started = time.monotonic()
        inp_sa = self._prepare(token_id, h_prev)
        x = gdn.rms_norm(inp_sa, self.vec("attn_norm.weight"))

        qg = self.runtime.matvec(self.view("attn_q.weight"), self.meta("attn_q.weight"), x)
        q, gate = attn.split_q_gate(qg)
        q = attn.rms_norm_heads(q, attn.N_HEAD, self.vec("attn_q_norm.weight"))
        k = self.runtime.matvec(self.view("attn_k.weight"), self.meta("attn_k.weight"), x)
        v = self.runtime.matvec(self.view("attn_v.weight"), self.meta("attn_v.weight"), x)
        k = attn.rms_norm_heads(k, attn.N_HEAD_KV, self.vec("attn_k_norm.weight"))
        q_rope = t2.rope_text_neox(q, attn.N_HEAD, position)
        k_rope = t2.rope_text_neox(k, attn.N_HEAD_KV, position)
        self.cache["k"].append(attn.f16_roundtrip(k_rope))
        self.cache["v"].append(attn.f16_roundtrip(v))

        qh = attn.split_heads(q_rope, attn.N_HEAD)
        pregate: list[float] = []
        n_ctx = len(self.cache["k"])
        for qidx in range(attn.N_HEAD):
            kvh = qidx // attn.GQA_REPEAT
            qv = qh[qidx]
            scores: list[float] = []
            for ti in range(n_ctx):
                kh = self.cache["k"][ti][kvh * attn.HEAD_DIM:(kvh + 1) * attn.HEAD_DIM]
                score = gen.f32(math.fsum(float(qv[d]) * float(kh[d]) for d in range(attn.HEAD_DIM)) * t2.SCALE_ATTN)
                scores.append(score)
            probs = gen.softmax_many(scores)
            for d in range(attn.HEAD_DIM):
                acc = gen.f32(0.0)
                for ti in range(n_ctx):
                    vv = self.cache["v"][ti][kvh * attn.HEAD_DIM + d]
                    acc = gen.addf(acc, gen.mulf(probs[ti], vv))
                pregate.append(acc)

        gs = [exact.sigmoid_f32(vv) for vv in gate]
        gated = [gen.mulf(pregate[i], gs[i]) for i in range(attn.Q_DIM)]
        ao = self.runtime.matvec(self.view("attn_output.weight"), self.meta("attn_output.weight"), gated)
        residual = [gen.addf(inp_sa[i], ao[i]) for i in range(gdn.HIDDEN)]
        post = gdn.rms_norm(residual, self.vec("post_attention_norm.weight"))
        fo = self._ffn(post)
        final = [gen.addf(residual[i], fo[i]) for i in range(gdn.HIDDEN)]

        h_nextn = gdn.rms_norm(final, self.vec("nextn.shared_head_norm.weight"))
        lm_started = time.monotonic()
        logits = base._stream_q8_logits(self.model, self.tensors["output.weight"], self.runtime, h_nextn)
        lm_seconds = time.monotonic() - lm_started
        top5 = base._topk(logits, 5)
        return {
            "token": int(top5[0]["token"]),
            "top5": top5,
            "h_nextn": h_nextn,
            "elapsed_seconds": time.monotonic() - started,
            "lm_head_seconds": lm_seconds,
        }

    def report(self) -> dict[str, Any]:
        return {
            "cache_positions": len(self.cache["k"]),
            "resident_mtp_weight_bytes": self.resident_weight_bytes,
            "quant": self.runtime.report(),
        }


def run(args) -> dict[str, Any]:
    tokenizer = gen.load_tokenizer(args.tokenizer_json)
    rendered, prompt_ids = gen.encode_prompt(tokenizer, args.prompt, raw=args.raw_prompt)
    if not prompt_ids or len(prompt_ids) > args.max_prompt_tokens:
        raise RuntimeError(f"prompt token count {len(prompt_ids)} outside 1..{args.max_prompt_tokens}")

    engine = gen.StatefulK3Generator(
        args.model, args.target_native_lib, args.state_lib, args.inventory, args.work_dir)
    mtp = MTPBlock(args.model, args.target_native_lib, args.q4_native_lib)
    zero_h = [0.0] * gdn.HIDDEN
    prev_h = zero_h
    hidden = None
    catchup_seconds = 0.0
    started = time.monotonic()
    try:
        for pos, token_id in enumerate(prompt_ids):
            t0 = time.monotonic()
            mtp.catchup(token_id, prev_h, pos)
            catchup_seconds += time.monotonic() - t0
            hidden = engine.step(token_id)
            prev_h = gdn.rms_norm(hidden, engine.output_norm_w)
        assert hidden is not None

        target_logits = engine.logits(hidden)
        target_top5 = base._topk(target_logits, 5)
        first_token = int(target_top5[0]["token"])

        draft = mtp.draft(first_token, prev_h, engine.position)

        # The exact target itself is the oracle for the drafted next token.
        verified_hidden = engine.step(first_token)
        verify_logits = engine.logits(verified_hidden)
        verify_top5 = base._topk(verify_logits, 5)
        verify_token = int(verify_top5[0]["token"])
        accepted = int(draft["token"]) == verify_token

        payload = {
            "schema": "qwen38-mtp1-real-probe-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt": args.prompt,
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "first_target_token": first_token,
            "first_target_top5": target_top5,
            "mtp_draft_token": int(draft["token"]),
            "mtp_draft_top5": draft["top5"],
            "target_verify_token": verify_token,
            "target_verify_top5": verify_top5,
            "mtp1_accepted": accepted,
            "mtp_catchup_seconds": catchup_seconds,
            "mtp_draft_seconds": float(draft["elapsed_seconds"]),
            "mtp_lm_head_seconds": float(draft["lm_head_seconds"]),
            "mtp": mtp.report(),
            "target_state": engine.state_report(),
            "global_lm_head_bytes": int(mtp.tensors["output.weight"].nbytes),
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "prompt_token_ids": prompt_ids,
            "first_target_token": first_token,
            "mtp_draft_token": payload["mtp_draft_token"],
            "target_verify_token": verify_token,
            "mtp1_accepted": accepted,
            "mtp_catchup_seconds": catchup_seconds,
            "mtp_draft_seconds": payload["mtp_draft_seconds"],
            "mtp_lm_head_seconds": payload["mtp_lm_head_seconds"],
            "resident_mtp_weight_bytes": payload["mtp"]["resident_mtp_weight_bytes"],
            "q4_weight_bytes_executed": payload["mtp"]["quant"]["q4_weight_bytes"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print(f"QWEN38_MTP1_REAL_PROBE_PASS acceptance={int(accepted)}")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert MTP_LAYER == 64
    assert gdn.HIDDEN == 5120
    assert attn.N_HEAD == 24
    assert attn.N_HEAD_KV == 4
    assert attn.HEAD_DIM == 256
    assert attn.QG_DIM == 12288
    assert attn.KV_DIM == 1024
    print(json.dumps({
        "schema": "qwen38-mtp1-real-probe-sanity-v1",
        "status": "PASS",
        "mtp_layer": MTP_LAYER,
        "query_heads": attn.N_HEAD,
        "kv_heads": attn.N_HEAD_KV,
        "head_dim": attn.HEAD_DIM,
        "zero_h_bytes": gdn.HIDDEN * 4,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--target-native-lib", type=Path, required=True)
    r.add_argument("--q4-native-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--tokenizer-json", type=Path, required=True)
    r.add_argument("--prompt", default="Hi")
    r.add_argument("--raw-prompt", action="store_true")
    r.add_argument("--max-prompt-tokens", type=int, default=4)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
