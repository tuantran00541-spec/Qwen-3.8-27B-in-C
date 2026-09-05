#!/usr/bin/env python3
"""Compact sequential MTP-1 acceptance survey on the exact custom Qwen3.8 target.

This intentionally reuses one target context and one MTP context.  It measures
multiple consecutive draft/verify positions without multiplying prompts or CI
jobs.  The target remains the oracle; MTP cache advances with the real target
input token at every position even when its draft is rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
from qwen38_mtp1_real_probe import MTPBlock, rss_gib

EOS_IDS = gen.EOS_IDS


def run(args) -> dict:
    if args.steps < 1 or args.steps > 16:
        raise ValueError("steps must be in 1..16")

    tokenizer = gen.load_tokenizer(args.tokenizer_json)
    rendered, prompt_ids = gen.encode_prompt(tokenizer, args.prompt, raw=args.raw_prompt)
    if not prompt_ids or len(prompt_ids) > args.max_prompt_tokens:
        raise RuntimeError(f"prompt token count {len(prompt_ids)} outside 1..{args.max_prompt_tokens}")

    engine = gen.StatefulK3Generator(
        args.model, args.target_native_lib, args.state_lib, args.inventory, args.work_dir)
    mtp = MTPBlock(args.model, args.target_native_lib, args.q4_native_lib)

    prev_h = [0.0] * gdn.HIDDEN
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

        t0 = time.monotonic()
        logits = engine.logits(hidden)
        initial_lm_seconds = time.monotonic() - t0
        initial_top5 = base._topk(logits, 5)
        current_token = int(initial_top5[0]["token"])

        records = []
        accepted = 0
        target_tokens = [current_token]
        mtp_seconds_total = 0.0
        mtp_lm_seconds_total = 0.0
        target_step_seconds_total = 0.0
        target_lm_seconds_total = 0.0

        for index in range(args.steps):
            position = engine.position
            draft = mtp.draft(current_token, prev_h, position)
            mtp_seconds_total += float(draft["elapsed_seconds"])
            mtp_lm_seconds_total += float(draft["lm_head_seconds"])

            t_step = time.monotonic()
            verified_hidden = engine.step(current_token)
            target_step_seconds = time.monotonic() - t_step
            target_step_seconds_total += target_step_seconds

            t_lm = time.monotonic()
            verify_logits = engine.logits(verified_hidden)
            target_lm_seconds = time.monotonic() - t_lm
            target_lm_seconds_total += target_lm_seconds
            verify_top5 = base._topk(verify_logits, 5)
            verify_token = int(verify_top5[0]["token"])
            is_accepted = int(draft["token"]) == verify_token
            accepted += int(is_accepted)

            records.append({
                "index": index,
                "position": position,
                "input_target_token": current_token,
                "mtp_draft_token": int(draft["token"]),
                "target_verify_token": verify_token,
                "accepted": is_accepted,
                "mtp_draft_top5": draft["top5"],
                "target_verify_top5": verify_top5,
                "mtp_draft_seconds": float(draft["elapsed_seconds"]),
                "mtp_lm_head_seconds": float(draft["lm_head_seconds"]),
                "target_step_seconds": target_step_seconds,
                "target_lm_head_seconds": target_lm_seconds,
            })

            prev_h = gdn.rms_norm(verified_hidden, engine.output_norm_w)
            current_token = verify_token
            target_tokens.append(current_token)
            if current_token in EOS_IDS:
                break

        completed = len(records)
        rate = accepted / completed if completed else 0.0
        target_state = engine.state_report()
        payload = {
            "schema": "qwen38-mtp1-sequence-survey-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt": args.prompt,
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "requested_steps": args.steps,
            "completed_steps": completed,
            "first_target_token": target_tokens[0],
            "first_target_top5": initial_top5,
            "target_token_ids": target_tokens,
            "target_text": tokenizer.decode(target_tokens, skip_special_tokens=False),
            "accepted_count": accepted,
            "acceptance_rate": rate,
            "records": records,
            "mtp_catchup_seconds": catchup_seconds,
            "initial_target_lm_head_seconds": initial_lm_seconds,
            "mtp_draft_seconds_total": mtp_seconds_total,
            "mtp_lm_head_seconds_total": mtp_lm_seconds_total,
            "target_step_seconds_total": target_step_seconds_total,
            "target_lm_head_seconds_total": target_lm_seconds_total,
            "mtp": mtp.report(),
            "target_state": target_state,
            "global_lm_head_bytes": int(mtp.tensors["output.weight"].nbytes),
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "completed_steps": completed,
            "target_token_ids": target_tokens,
            "target_text": payload["target_text"],
            "accepted_count": accepted,
            "acceptance_rate": rate,
            "mtp_draft_seconds_total": mtp_seconds_total,
            "mtp_lm_head_seconds_total": mtp_lm_seconds_total,
            "target_step_seconds_total": target_step_seconds_total,
            "target_lm_head_seconds_total": target_lm_seconds_total,
            "target_k3_bytes_read": target_state["reader"]["bytes_read"],
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2, ensure_ascii=False))
        for r in records:
            print(
                "MTP1_SURVEY_STEP",
                r["index"],
                "pos", r["position"],
                "in", r["input_target_token"],
                "draft", r["mtp_draft_token"],
                "verify", r["target_verify_token"],
                "accepted", int(r["accepted"]),
            )
        print(f"QWEN38_MTP1_SEQUENCE_SURVEY_PASS accepted={accepted}/{completed}")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert EOS_IDS == {248044, 248046}
    print(json.dumps({
        "schema": "qwen38-mtp1-sequence-survey-sanity-v1",
        "status": "PASS",
        "max_steps": 16,
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
    r.add_argument("--steps", type=int, default=6)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
