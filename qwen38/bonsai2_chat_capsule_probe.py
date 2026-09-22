#!/usr/bin/env python3
"""Real-weight save/close/load/continue parity probe for chat-history capsules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import bonsai2_chat_capsule as capsule
import bonsai2_prompt_spike as prompt
import qwen35_k3_generate as textgen


def feed_tokens(engine, token_ids, *, tail_tokens: int):
    hidden = None
    for token_id in token_ids:
        engine.trim_attention_kv(max(0, tail_tokens - 1))
        hidden = engine.step(int(token_id))
    return hidden


def generate(engine, tokenizer, hidden, *, tail_tokens: int, max_new_tokens: int):
    token_ids = []
    for index in range(max_new_tokens):
        logits = engine.logits(hidden)
        token_id = int(prompt.base.topk(logits, 1)[0]["token"])
        token_ids.append(token_id)
        if token_id in prompt.EOS_IDS or index + 1 >= max_new_tokens:
            break
        engine.trim_attention_kv(max(0, tail_tokens - 1))
        hidden = engine.step(token_id)
    return token_ids, tokenizer.decode(token_ids, skip_special_tokens=False)


def make_engine(args, work_dir: Path):
    return prompt.StatefulBonsai2Generator(
        args.model,
        args.native_lib,
        args.state_lib,
        work_dir,
        args.threads,
        resident_decoder=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--native-lib", type=Path, required=True)
    ap.add_argument("--state-lib", type=Path, required=True)
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tail", type=int, default=31)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = textgen.load_tokenizer(args.tokenizer_json)
    initial = (
        "Memorize this conversation state. The codeword is MARBLE-731 and "
        "the project number is 17. We are testing whether a saved runtime "
        "state can resume the same conversation after the engine is closed. "
        "Keep these facts available for the next turn."
    )
    rendered, initial_ids = textgen.encode_prompt(tokenizer, initial, raw=False)
    suffix_text = (
        "\n<|im_start|>user\nReply briefly with the remembered codeword and "
        "project number.<|im_end|>\n<|im_start|>assistant\n"
    )
    suffix_ids = list(tokenizer.encode(suffix_text, add_special_tokens=False).ids)

    capsule_path = args.work_dir / "chat-state.q38cap"

    engine = make_engine(args, args.work_dir / "baseline-k3")
    try:
        for token_id in initial_ids:
            engine.step(int(token_id))
        meta = capsule.save_chat_capsule(
            engine, capsule_path, attention_tail_tokens=args.tail
        )
        checkpoint_fp = capsule.engine_fingerprint(engine)
        baseline_hidden = feed_tokens(engine, suffix_ids, tail_tokens=args.tail)
        baseline_ids, baseline_text = generate(
            engine,
            tokenizer,
            baseline_hidden,
            tail_tokens=args.tail,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        engine.close()

    restored = make_engine(args, args.work_dir / "restored-k3")
    try:
        restored_meta = capsule.load_chat_capsule(restored, capsule_path)
        restored_fp = capsule.engine_fingerprint(restored)
        restored_hidden = feed_tokens(restored, suffix_ids, tail_tokens=args.tail)
        restored_ids, restored_text = generate(
            restored,
            tokenizer,
            restored_hidden,
            tail_tokens=args.tail,
            max_new_tokens=args.max_new_tokens,
        )
        restored_state = restored.report()
    finally:
        restored.close()

    exact = baseline_ids == restored_ids and baseline_text == restored_text
    result = {
        "schema": "qwen38-bonsai2-chat-capsule-probe-v1",
        "status": "PASS" if exact else "FAIL",
        "tail_tokens": args.tail,
        "initial_prompt_tokens": len(initial_ids),
        "followup_tokens": len(suffix_ids),
        "checkpoint_fingerprint": checkpoint_fp,
        "restored_fingerprint": restored_fp,
        "fingerprint_match": checkpoint_fp == restored_fp,
        "generated_token_ids_baseline": baseline_ids,
        "generated_token_ids_restored": restored_ids,
        "generated_text_baseline": baseline_text,
        "generated_text_restored": restored_text,
        "exact_resume_parity": exact,
        "capsule_file_bytes": meta["capsule_file_bytes"],
        "capsule_payload_bytes": meta["payload_bytes"],
        "capsule_position": meta["position"],
        "kept_prior_attention_rows": meta["kept_prior_attention_rows"],
        "restored_attention_kv_bytes_f16": restored_state["attention_kv_bytes_f16"],
        "model_sha256": prompt.base.MODEL_SHA256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not exact:
        raise SystemExit(1)
    print("QWEN38_BONSAI2_CHAT_CAPSULE_RESUME_PASS")


if __name__ == "__main__":
    main()
