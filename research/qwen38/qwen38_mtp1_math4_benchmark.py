#!/usr/bin/env python3
"""Bounded four-problem math benchmark for the exact Qwen3.8 SSD runtime.

The prompt is intentionally compact so the benchmark measures the newly proven
staged multi-vector prefill rather than wasting TTFT on verbose instructions.
Target prompt prefill uses the exact layer-major + native-F32 + matvec-many path.
MTP prompt catch-up remains token-serial, then generation uses the already
proven transactional MTP-1 target verifier.

Math correctness is reported separately from runtime correctness.  A wrong or
truncated model answer does not turn the runtime workflow red; it receives a
0..4 score and the raw output is retained as evidence.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import resource
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
import qwen38_k3_pair_transaction_probe as tx
import qwen38_lm_head_pair_probe as lm_pair
import qwen38_k3_prompt_block_many_probe as prefill_many
from native_f32_runtime import enable_native_f32
from quant_many_runtime import enable_quant_many
from qwen38_mtp1_real_probe import MTPBlock

EOS_IDS = gen.EOS_IDS
LM_HEAD_BYTES = 1_350_860_800
K3_STREAM_BYTES = 21_127_430_144

MATH4_PROMPT = """Return exactly 4 final answers, no reasoning.
1) sqrt(x+6)+sqrt(x-3)=5,x>=3
2) Urn 5R4B3G; draw 3 without replacement: P(exactly 2 colors)?
3) 7^2026 mod 1000
4) positive integers a<=b: 1/a+1/b=1/6; list all pairs"""


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _top1(logits) -> int:
    return int(base._topk(logits, 1)[0]["token"])


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def grade_math4(text: str) -> dict[str, Any]:
    c = _compact(text)
    q1 = ("139/25" in c) or ("5.56" in c)
    q2 = "29/44" in c
    q3 = bool(re.search(r"(?:^|[^0-9])649(?:[^0-9]|$)", c))
    pairs = ["(7,42)", "(8,24)", "(9,18)", "(10,15)", "(12,12)"]
    q4 = all(p in c for p in pairs)
    checks = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
    return {
        "score": sum(int(v) for v in checks.values()),
        "out_of": 4,
        "checks": checks,
        "reference_answers": {
            "q1": "139/25 (5.56)",
            "q2": "29/44",
            "q3": "649",
            "q4": pairs,
        },
        "grading_note": "Heuristic exact-answer presence check; runtime PASS is independent of math score.",
    }


def run(args) -> dict[str, Any]:
    if args.max_new_tokens < 8 or args.max_new_tokens > 64:
        raise ValueError("max-new-tokens must be in 8..64")

    tokenizer_started = time.monotonic()
    tokenizer = gen.load_tokenizer(args.tokenizer_json)
    rendered, prompt_ids = gen.encode_prompt(tokenizer, MATH4_PROMPT, raw=False)
    tokenizer_seconds = time.monotonic() - tokenizer_started
    if not prompt_ids or len(prompt_ids) > args.max_prompt_tokens:
        raise RuntimeError(
            f"prompt token count {len(prompt_ids)} outside 1..{args.max_prompt_tokens}")

    t0 = time.monotonic()
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    engine_init_seconds = time.monotonic() - t0
    try:
        native = enable_native_f32(engine, args.f32_lib)
        many = enable_quant_many(engine, args.many_lib)

        t0 = time.monotonic()
        mtp = MTPBlock(args.model, args.quant_lib, args.q4_native_lib)
        mtp_init_seconds = time.monotonic() - t0
        runtime_init_seconds = engine_init_seconds + mtp_init_seconds

        tensor = engine.tensors["output.weight"]
        if int(tensor.nbytes) != LM_HEAD_BYTES:
            raise RuntimeError(f"unexpected LM-head bytes: {tensor.nbytes}")

        # ----- Staged exact prompt prefill / TTFT -----
        prompt_started = time.monotonic()
        prompt_reader_before = int(engine.reader.report()["bytes_read"])
        native_before = native.native_f32_calls
        many_before = many.many_calls
        many_vectors_before = many.many_vectors

        t0 = time.monotonic()
        hidden_rows = prefill_many.step_block_many(engine, prompt_ids)
        prompt_target_seconds = time.monotonic() - t0
        if len(hidden_rows) != len(prompt_ids):
            raise RuntimeError("staged prefill hidden-row count mismatch")
        prompt_k3_bytes = int(engine.reader.report()["bytes_read"]) - prompt_reader_before
        if prompt_k3_bytes != K3_STREAM_BYTES:
            raise RuntimeError(f"staged prompt prefill must use one K3 stream, got {prompt_k3_bytes}")

        # MTP catch-up consumes the target hidden from the previous position.
        prev_h = [0.0] * gdn.HIDDEN
        prompt_mtp_catchup_seconds = 0.0
        for pos, token_id in enumerate(prompt_ids):
            t0 = time.monotonic()
            mtp.catchup(int(token_id), prev_h, pos)
            prompt_mtp_catchup_seconds += time.monotonic() - t0
            prev_h = gdn.rms_norm(hidden_rows[pos], engine.output_norm_w)
        hidden = hidden_rows[-1]

        t0 = time.monotonic()
        first_logits, first_head_bytes = lm_pair._stream_one_counted(
            args.model, tensor, engine.runtime, hidden, engine.output_norm_w)
        initial_lm_seconds = time.monotonic() - t0
        first_token = _top1(first_logits)
        ttft_seconds = time.monotonic() - prompt_started

        # ----- Bounded MTP-1 generation -----
        output_tokens = [first_token]
        current = first_token
        generation_started = time.monotonic()
        generation_reader_before = int(engine.reader.report()["bytes_read"])
        records: list[dict[str, Any]] = []
        accepted_count = 0
        speculative_rounds = 0
        target_pair_seconds_total = 0.0
        target_lm_seconds_total = 0.0
        mtp_draft_seconds_total = 0.0
        mtp_lm_seconds_total = 0.0
        mtp_catchup_seconds_total = 0.0
        rollback_seconds_total = 0.0
        generation_target_lm_bytes = 0
        generation_mtp_lm_bytes = 0

        while len(output_tokens) < args.max_new_tokens and current not in EOS_IDS:
            remaining = args.max_new_tokens - len(output_tokens)
            position = int(engine.position)

            if remaining == 1:
                t0 = time.monotonic()
                mtp.catchup(current, prev_h, position)
                mtp_catchup_seconds_total += time.monotonic() - t0

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
                records.append({
                    "mode": "single-tail",
                    "position": position,
                    "target_next": next_token,
                    "target_seconds": target_seconds,
                    "target_lm_seconds": head_seconds,
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

            t0 = time.monotonic()
            logits_a, logits_b, pair_head_bytes = lm_pair._stream_pair_counted(
                args.model, tensor, engine.runtime, hidden_a, hidden_b, engine.output_norm_w)
            pair_head_seconds = time.monotonic() - t0
            target_lm_seconds_total += pair_head_seconds
            generation_target_lm_bytes += int(pair_head_bytes)

            verify_a = _top1(logits_a)
            verify_b = _top1(logits_b)
            accepted = draft_token == verify_a
            accepted_count += int(accepted)
            norm_a = gdn.rms_norm(hidden_a, engine.output_norm_w)

            record = {
                "mode": "speculative-pair",
                "position": position,
                "input_token": int(current),
                "mtp_draft": draft_token,
                "target_verify_a": verify_a,
                "target_verify_b": verify_b,
                "accepted": accepted,
                "pair_k3_bytes": int(pair_k3_bytes),
                "pair_seconds": pair_seconds,
                "target_pair_lm_seconds": pair_head_seconds,
                "mtp_draft_seconds": float(draft["elapsed_seconds"]),
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
                if verify_a in EOS_IDS:
                    current = verify_a
                    records.append(record)
                    break
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

            records.append(record)

        generation_seconds = time.monotonic() - generation_started
        generation_k3_bytes = int(engine.reader.report()["bytes_read"]) - generation_reader_before
        transitions = max(0, len(output_tokens) - 1)
        tps = transitions / generation_seconds if generation_seconds > 0 else 0.0
        spt = generation_seconds / transitions if transitions else None
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)
        acceptance_rate = accepted_count / speculative_rounds if speculative_rounds else 0.0
        cache_aligned = len(mtp.cache["k"]) == int(engine.position)
        if not cache_aligned:
            raise RuntimeError(
                f"MTP/target position mismatch: mtp={len(mtp.cache['k'])} target={engine.position}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("math benchmark requires direct I/O")

        grade = grade_math4(output_text)
        payload = {
            "schema": "qwen38-mtp1-math4-benchmark-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prompt": MATH4_PROMPT,
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "requested_new_tokens": args.max_new_tokens,
            "generated_token_ids": output_tokens,
            "generated_token_count": len(output_tokens),
            "generated_text": output_text,
            "stopped_on_eos": bool(output_tokens and output_tokens[-1] in EOS_IDS),
            "math_grade": grade,
            "startup": {
                "tokenizer_seconds": tokenizer_seconds,
                "engine_init_seconds": engine_init_seconds,
                "mtp_init_seconds": mtp_init_seconds,
                "runtime_init_seconds": runtime_init_seconds,
                "cold_start_to_first_token_seconds": runtime_init_seconds + ttft_seconds,
            },
            "ttft": {
                "seconds_after_runtime_init": ttft_seconds,
                "prompt_target_staged_prefill_seconds": prompt_target_seconds,
                "prompt_mtp_catchup_seconds": prompt_mtp_catchup_seconds,
                "initial_lm_head_seconds": initial_lm_seconds,
                "initial_lm_head_bytes": int(first_head_bytes),
                "prompt_k3_bytes": prompt_k3_bytes,
                "native_f32_calls": native.native_f32_calls - native_before,
                "many_calls": many.many_calls - many_before,
                "many_vectors": many.many_vectors - many_vectors_before,
            },
            "generation": {
                "seconds": generation_seconds,
                "transitions_after_first_token": transitions,
                "tokens_per_second_after_first_token": tps,
                "seconds_per_token_after_first_token": spt,
                "speculative_rounds": speculative_rounds,
                "accepted_rounds": accepted_count,
                "acceptance_rate": acceptance_rate,
                "target_pair_seconds_total": target_pair_seconds_total,
                "target_lm_seconds_total": target_lm_seconds_total,
                "mtp_draft_seconds_total": mtp_draft_seconds_total,
                "mtp_lm_head_seconds_total": mtp_lm_seconds_total,
                "mtp_catchup_seconds_total": mtp_catchup_seconds_total,
                "rollback_seconds_total": rollback_seconds_total,
                "k3_bytes": generation_k3_bytes,
                "effective_k3_bytes_per_transition": generation_k3_bytes / transitions if transitions else None,
                "target_lm_head_bytes": generation_target_lm_bytes,
                "mtp_lm_head_bytes": generation_mtp_lm_bytes,
                "records": records,
            },
            "mtp_target_cache_aligned": cache_aligned,
            "target_reader": engine.reader.report(),
            "native_f32": native.report(),
            "quant_many": many.report(),
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "prompt_token_count": len(prompt_ids),
            "generated_token_count": len(output_tokens),
            "generated_text": output_text,
            "math_score": grade["score"],
            "math_checks": grade["checks"],
            "runtime_init_seconds": runtime_init_seconds,
            "ttft_seconds_after_runtime_init": ttft_seconds,
            "staged_prefill_seconds": prompt_target_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second_after_first_token": tps,
            "seconds_per_token_after_first_token": spt,
            "speculative_rounds": speculative_rounds,
            "accepted_rounds": accepted_count,
            "acceptance_rate": acceptance_rate,
            "prompt_k3_bytes": prompt_k3_bytes,
            "generation_k3_bytes": generation_k3_bytes,
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2, ensure_ascii=False))
        print(f"QWEN38_MTP1_MATH4_BENCHMARK_PASS score={grade['score']}/4")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    g = grade_math4("1) 139/25; 2) 29/44; 3) 649; 4) (7,42),(8,24),(9,18),(10,15),(12,12)")
    if g["score"] != 4:
        raise RuntimeError("math4 grader sanity failed")
    print(json.dumps({
        "schema": "qwen38-mtp1-math4-benchmark-sanity-v1",
        "status": "PASS",
        "prompt": MATH4_PROMPT,
        "reference_score": g["score"],
        "prefill": "staged exact layer-major native-F32 + matvec-many",
        "generation": "transactional MTP-1",
    }, indent=2, ensure_ascii=False))


def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    runp = sub.add_parser("run")
    runp.add_argument("--model", type=Path, required=True)
    runp.add_argument("--quant-lib", type=Path, required=True)
    runp.add_argument("--many-lib", type=Path, required=True)
    runp.add_argument("--q4-native-lib", type=Path, required=True)
    runp.add_argument("--state-lib", type=Path, required=True)
    runp.add_argument("--f32-lib", type=Path, required=True)
    runp.add_argument("--inventory", type=Path, required=True)
    runp.add_argument("--tokenizer-json", type=Path, required=True)
    runp.add_argument("--max-prompt-tokens", type=int, default=96)
    runp.add_argument("--max-new-tokens", type=int, default=40)
    runp.add_argument("--work-dir", type=Path, required=True)
    runp.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "sanity":
        sanity()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
