#!/usr/bin/env python3
"""Bounded real-prompt MTP-1 benchmark for the exact Qwen3.8 SSD runtime.

This is a performance benchmark, not a new correctness proof.  Correctness is
anchored by the preceding exact miss/hit end-to-end gate.  Here we use a real
instruction that asks for a long essay, generate a bounded prefix so hosted CI
finishes in a useful time, and report measured TTFT/steady-state throughput.
Any 700/1000-word completion times are explicitly projections from the measured
prefix, not claims that the full essay was generated in CI.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
import qwen38_k3_pair_transaction_probe as tx
import qwen38_lm_head_pair_probe as lm_pair
from qwen38_mtp1_real_probe import MTPBlock

EOS_IDS = gen.EOS_IDS
LM_HEAD_BYTES = 1_350_860_800
K3_STREAM_BYTES = 21_127_430_144


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _top1(logits) -> int:
    return int(base._topk(logits, 1)[0]["token"])


def _projection(tokens_per_second: float, ttft: float, words: int) -> dict[str, Any]:
    # English prose is often roughly 1.2-1.5 model tokens/word.  Keep this as a
    # range instead of pretending the short prefix establishes a precise ratio.
    lo_tokens = int(math.ceil(words * 1.2))
    hi_tokens = int(math.ceil(words * 1.5))
    if tokens_per_second <= 0.0:
        return {"words": words, "status": "unavailable"}
    lo_gen = lo_tokens / tokens_per_second
    hi_gen = hi_tokens / tokens_per_second
    return {
        "words": words,
        "assumed_tokens_per_word": [1.2, 1.5],
        "projected_output_tokens": [lo_tokens, hi_tokens],
        "projected_generation_seconds": [lo_gen, hi_gen],
        "projected_total_after_runtime_init_seconds": [ttft + lo_gen, ttft + hi_gen],
        "projection_only": True,
    }


def run(args) -> dict[str, Any]:
    if args.max_new_tokens < 4 or args.max_new_tokens > 32:
        raise ValueError("max-new-tokens must be in 4..32")

    tokenizer_started = time.monotonic()
    tokenizer = gen.load_tokenizer(args.tokenizer_json)
    rendered, prompt_ids = gen.encode_prompt(tokenizer, args.prompt, raw=args.raw_prompt)
    tokenizer_seconds = time.monotonic() - tokenizer_started
    if not prompt_ids or len(prompt_ids) > args.max_prompt_tokens:
        raise RuntimeError(
            f"prompt token count {len(prompt_ids)} outside 1..{args.max_prompt_tokens}")

    t0 = time.monotonic()
    engine = gen.StatefulK3Generator(
        args.model, args.target_native_lib, args.state_lib, args.inventory, args.work_dir)
    engine_init_seconds = time.monotonic() - t0
    try:
        t0 = time.monotonic()
        mtp = MTPBlock(args.model, args.target_native_lib, args.q4_native_lib)
        mtp_init_seconds = time.monotonic() - t0
        runtime_init_seconds = engine_init_seconds + mtp_init_seconds

        tensor = engine.tensors["output.weight"]
        if int(tensor.nbytes) != LM_HEAD_BYTES:
            raise RuntimeError(f"unexpected LM-head bytes: {tensor.nbytes}")

        # ----- Prompt ingest / TTFT -----
        prompt_started = time.monotonic()
        prompt_reader_before = int(engine.reader.report()["bytes_read"])
        prev_h = [0.0] * gdn.HIDDEN
        hidden = None
        prompt_target_seconds = 0.0
        prompt_mtp_catchup_seconds = 0.0
        for pos, token_id in enumerate(prompt_ids):
            t0 = time.monotonic()
            mtp.catchup(int(token_id), prev_h, pos)
            prompt_mtp_catchup_seconds += time.monotonic() - t0
            t0 = time.monotonic()
            hidden = engine.step(int(token_id))
            prompt_target_seconds += time.monotonic() - t0
            prev_h = gdn.rms_norm(hidden, engine.output_norm_w)
        assert hidden is not None
        prompt_k3_bytes = int(engine.reader.report()["bytes_read"]) - prompt_reader_before

        t0 = time.monotonic()
        first_logits, first_head_bytes = lm_pair._stream_one_counted(
            args.model, tensor, engine.runtime, hidden, engine.output_norm_w)
        initial_lm_seconds = time.monotonic() - t0
        first_token = _top1(first_logits)
        ttft_seconds = time.monotonic() - prompt_started

        # ----- Bounded speculative generation -----
        output_tokens = [first_token]
        current = first_token
        generation_started = time.monotonic()
        generation_reader_before = int(engine.reader.report()["bytes_read"])
        records: list[dict[str, Any]] = []
        accepted_count = 0
        speculative_rounds = 0
        single_tail_steps = 0
        target_pair_seconds_total = 0.0
        target_lm_seconds_total = 0.0
        mtp_draft_seconds_total = 0.0
        mtp_lm_seconds_total = 0.0
        mtp_catchup_seconds_total = 0.0
        checkpoint_copy_seconds_total = 0.0
        rollback_seconds_total = 0.0
        generation_target_lm_bytes = 0
        generation_mtp_lm_bytes = 0

        while len(output_tokens) < args.max_new_tokens and current not in EOS_IDS:
            remaining = args.max_new_tokens - len(output_tokens)
            position = int(engine.position)

            # Avoid overshooting the requested benchmark length with a 2-token
            # accepted pair.  The final one-transition tail uses an ordinary
            # target step while keeping the MTP cache aligned.
            if remaining == 1:
                t0 = time.monotonic()
                mtp.catchup(current, prev_h, position)
                tail_mtp_catchup = time.monotonic() - t0
                mtp_catchup_seconds_total += tail_mtp_catchup

                t0 = time.monotonic()
                hidden_tail = engine.step(current)
                target_seconds = time.monotonic() - t0
                target_pair_seconds_total += target_seconds

                t0 = time.monotonic()
                logits_tail, head_bytes = lm_pair._stream_one_counted(
                    args.model, tensor, engine.runtime, hidden_tail, engine.output_norm_w)
                head_seconds = time.monotonic() - t0
                target_lm_seconds_total += head_seconds
                generation_target_lm_bytes += int(head_bytes)
                next_token = _top1(logits_tail)
                output_tokens.append(next_token)
                prev_h = gdn.rms_norm(hidden_tail, engine.output_norm_w)
                current = next_token
                single_tail_steps += 1
                records.append({
                    "mode": "single-tail",
                    "position": position,
                    "input_token": int(output_tokens[-2]),
                    "target_next": next_token,
                    "target_seconds": target_seconds,
                    "target_lm_seconds": head_seconds,
                    "target_lm_bytes": int(head_bytes),
                    "mtp_catchup_seconds": tail_mtp_catchup,
                })
                continue

            draft = mtp.draft(current, prev_h, position)
            speculative_rounds += 1
            mtp_draft_seconds_total += float(draft["elapsed_seconds"])
            mtp_lm_seconds_total += float(draft["lm_head_seconds"])
            generation_mtp_lm_bytes += LM_HEAD_BYTES
            draft_token = int(draft["token"])

            hidden_a, hidden_b, txn, pair_seconds, pair_k3_bytes = tx._run_pair_tx(
                engine, current, draft_token)
            target_pair_seconds_total += pair_seconds
            checkpoint_copy_seconds_total += float(txn["checkpoint_copy_seconds"])

            t0 = time.monotonic()
            logits_a, logits_b, pair_head_bytes = lm_pair._stream_pair_counted(
                args.model,
                tensor,
                engine.runtime,
                hidden_a,
                hidden_b,
                engine.output_norm_w,
            )
            pair_head_seconds = time.monotonic() - t0
            target_lm_seconds_total += pair_head_seconds
            generation_target_lm_bytes += int(pair_head_bytes)

            verify_a = _top1(logits_a)
            verify_b = _top1(logits_b)
            accepted = draft_token == verify_a
            accepted_count += int(accepted)
            norm_a = gdn.rms_norm(hidden_a, engine.output_norm_w)
            record: dict[str, Any] = {
                "mode": "speculative-pair",
                "position": position,
                "input_token": int(current),
                "mtp_draft": draft_token,
                "target_verify_a": verify_a,
                "target_verify_b": verify_b,
                "accepted": accepted,
                "pair_k3_bytes": int(pair_k3_bytes),
                "pair_seconds": pair_seconds,
                "target_pair_lm_bytes": int(pair_head_bytes),
                "target_pair_lm_seconds": pair_head_seconds,
                "mtp_draft_seconds": float(draft["elapsed_seconds"]),
                "mtp_lm_head_seconds": float(draft["lm_head_seconds"]),
                "checkpoint_copy_seconds": float(txn["checkpoint_copy_seconds"]),
            }

            if accepted:
                output_tokens.append(verify_a)
                t0 = time.monotonic()
                mtp.catchup(draft_token, norm_a, position + 1)
                hit_catchup = time.monotonic() - t0
                mtp_catchup_seconds_total += hit_catchup
                prev_h = gdn.rms_norm(hidden_b, engine.output_norm_w)
                record["mtp_hit_catchup_seconds"] = hit_catchup
                record["rollback_seconds"] = 0.0
                record["committed_target_positions"] = 2

                if verify_a in EOS_IDS:
                    current = verify_a
                    records.append(record)
                    break

                # remaining >= 2 by construction, so a normal hit may emit C.
                output_tokens.append(verify_b)
                current = verify_b
                if verify_b in EOS_IDS:
                    records.append(record)
                    break
            else:
                rollback_seconds = tx.rollback_second(engine, txn)
                rollback_seconds_total += rollback_seconds
                output_tokens.append(verify_a)
                prev_h = norm_a
                current = verify_a
                record["mtp_hit_catchup_seconds"] = 0.0
                record["rollback_seconds"] = rollback_seconds
                record["committed_target_positions"] = 1

            records.append(record)

        generation_seconds = time.monotonic() - generation_started
        generation_k3_bytes = int(engine.reader.report()["bytes_read"]) - generation_reader_before
        transitions = max(0, len(output_tokens) - 1)
        tokens_per_second = transitions / generation_seconds if generation_seconds > 0 else 0.0
        seconds_per_token = generation_seconds / transitions if transitions else None
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)
        output_word_count = len(output_text.split())
        acceptance_rate = accepted_count / speculative_rounds if speculative_rounds else 0.0
        total_generation_lm_bytes = generation_target_lm_bytes + generation_mtp_lm_bytes

        # MTP cache should cover exactly the positions already processed by the
        # target.  The last emitted token remains the unprocessed current token.
        cache_aligned = len(mtp.cache["k"]) == int(engine.position)
        if not cache_aligned:
            raise RuntimeError(
                f"MTP/target position mismatch: mtp={len(mtp.cache['k'])} target={engine.position}")
        if prompt_k3_bytes != len(prompt_ids) * K3_STREAM_BYTES:
            raise RuntimeError(f"unexpected prompt K3 traffic: {prompt_k3_bytes}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("essay benchmark requires direct I/O")

        payload = {
            "schema": "qwen38-mtp1-essay-benchmark-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "benchmark_scope": "bounded-prefix; full essay not generated",
            "prompt": args.prompt,
            "rendered_prompt": rendered,
            "raw_prompt": bool(args.raw_prompt),
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "requested_new_tokens": args.max_new_tokens,
            "generated_token_ids": output_tokens,
            "generated_token_count": len(output_tokens),
            "generated_text_prefix": output_text,
            "generated_prefix_word_count": output_word_count,
            "stopped_on_eos": bool(output_tokens and output_tokens[-1] in EOS_IDS),
            "startup": {
                "tokenizer_seconds": tokenizer_seconds,
                "engine_init_seconds": engine_init_seconds,
                "mtp_init_seconds": mtp_init_seconds,
                "runtime_init_seconds": runtime_init_seconds,
                "cold_start_to_first_token_seconds": runtime_init_seconds + ttft_seconds,
            },
            "ttft": {
                "seconds_after_runtime_init": ttft_seconds,
                "prompt_target_seconds": prompt_target_seconds,
                "prompt_mtp_catchup_seconds": prompt_mtp_catchup_seconds,
                "initial_lm_head_seconds": initial_lm_seconds,
                "initial_lm_head_bytes": int(first_head_bytes),
                "prompt_k3_bytes": prompt_k3_bytes,
            },
            "generation": {
                "seconds": generation_seconds,
                "transitions_after_first_token": transitions,
                "tokens_per_second_after_first_token": tokens_per_second,
                "seconds_per_token_after_first_token": seconds_per_token,
                "speculative_rounds": speculative_rounds,
                "accepted_rounds": accepted_count,
                "acceptance_rate": acceptance_rate,
                "single_tail_steps": single_tail_steps,
                "target_pair_seconds_total": target_pair_seconds_total,
                "target_lm_seconds_total": target_lm_seconds_total,
                "mtp_draft_seconds_total": mtp_draft_seconds_total,
                "mtp_lm_head_seconds_total": mtp_lm_seconds_total,
                "mtp_catchup_seconds_total": mtp_catchup_seconds_total,
                "checkpoint_copy_seconds_total": checkpoint_copy_seconds_total,
                "rollback_seconds_total": rollback_seconds_total,
                "k3_bytes": generation_k3_bytes,
                "effective_k3_bytes_per_transition": generation_k3_bytes / transitions if transitions else None,
                "target_lm_head_bytes": generation_target_lm_bytes,
                "mtp_lm_head_bytes": generation_mtp_lm_bytes,
                "total_lm_head_bytes": total_generation_lm_bytes,
                "decoder_plus_lm_head_bytes": generation_k3_bytes + total_generation_lm_bytes,
                "records": records,
            },
            "projection": {
                "note": "Projection only; CI did not generate the full essay.",
                "word_to_token_assumption": "1.2-1.5 tokens/word for English prose",
                "essay_700_words": _projection(tokens_per_second, ttft_seconds, 700),
                "essay_1000_words": _projection(tokens_per_second, ttft_seconds, 1000),
            },
            "mtp": mtp.report(),
            "target_reader": engine.reader.report(),
            "mtp_target_cache_aligned": cache_aligned,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "prompt": args.prompt,
            "prompt_token_count": len(prompt_ids),
            "generated_token_count": len(output_tokens),
            "generated_text_prefix": output_text,
            "runtime_init_seconds": runtime_init_seconds,
            "ttft_seconds_after_runtime_init": ttft_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second_after_first_token": tokens_per_second,
            "seconds_per_token_after_first_token": seconds_per_token,
            "speculative_rounds": speculative_rounds,
            "accepted_rounds": accepted_count,
            "acceptance_rate": acceptance_rate,
            "generation_k3_bytes": generation_k3_bytes,
            "effective_k3_bytes_per_transition": payload["generation"]["effective_k3_bytes_per_transition"],
            "max_rss_gib": payload["max_rss_gib"],
            "projection_700_words_seconds": payload["projection"]["essay_700_words"].get("projected_total_after_runtime_init_seconds"),
            "projection_1000_words_seconds": payload["projection"]["essay_1000_words"].get("projected_total_after_runtime_init_seconds"),
        }, indent=2, ensure_ascii=False))
        print("QWEN38_MTP1_ESSAY_BENCHMARK_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert EOS_IDS == {248044, 248046}
    assert LM_HEAD_BYTES == 1_350_860_800
    assert K3_STREAM_BYTES == 21_127_430_144
    print(json.dumps({
        "schema": "qwen38-mtp1-essay-benchmark-sanity-v1",
        "status": "PASS",
        "benchmark": "bounded real prompt; full essay projected only",
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
    r.add_argument("--prompt", default="Write a 700-word essay on curiosity.")
    r.add_argument("--raw-prompt", action="store_true")
    r.add_argument("--max-prompt-tokens", type=int, default=16)
    r.add_argument("--max-new-tokens", type=int, default=12)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
