#!/usr/bin/env python3
"""Exact end-to-end MTP-1 speculative decoding probe for Qwen3.8-27B.

This is an isolated research gate.  It combines the independently proven
primitives without changing the default generator:

- native blk.64 MTP-1 draft;
- two target positions per K3 weight stream;
- transactional rollback of the second target position on a miss;
- two target LM-head vectors per one output-weight stream.

The compact oracle is deliberately chosen to exercise both branches in one
sequence.  From raw token 12675 ("Hi") the exact target emits:

    11 -> 353 -> 2688 -> 264

MTP misses on 11 (draft 1092, target 353), then hits on 353 (draft 2688).
Thus three ordinary target steps must collapse to two speculative K3 streams.
The gate requires identical emitted tokens and bitwise-identical final target
GDN/conv/attention state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
import qwen38_k3_pair_reuse_probe as reuse
import qwen38_k3_pair_transaction_probe as tx
import qwen38_lm_head_pair_probe as lm_pair
from qwen38_mtp1_real_probe import MTPBlock

PREFIX_TOKEN = 12675
EXPECTED_TOKENS = [11, 353, 2688, 264]
EXPECTED_BRANCHES = [False, True]
TRANSITIONS = 3


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _top1(logits) -> int:
    return int(base._topk(logits, 1)[0]["token"])


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.target_native_lib, args.state_lib, args.inventory, args.work_dir)
    mtp = MTPBlock(args.model, args.target_native_lib, args.q4_native_lib)
    started = time.monotonic()
    try:
        # Shared prefix.  MTP and target must see the same token/state history.
        zero_h = [0.0] * gdn.HIDDEN
        mtp_catchup_started = time.monotonic()
        mtp.catchup(PREFIX_TOKEN, zero_h, 0)
        prefix_mtp_catchup_seconds = time.monotonic() - mtp_catchup_started

        prefix_hidden = engine.step(PREFIX_TOKEN)
        prefix_h = gdn.rms_norm(prefix_hidden, engine.output_norm_w)
        initial_lm_started = time.monotonic()
        first_logits = engine.logits(prefix_hidden)
        initial_lm_seconds = time.monotonic() - initial_lm_started
        first_token = _top1(first_logits)
        if first_token != EXPECTED_TOKENS[0]:
            raise RuntimeError(f"pinned first token changed: {first_token}")

        prefix_snap = reuse.capture_state(engine)

        # ---- Ordinary exact baseline: three target steps. ----
        baseline_started = time.monotonic()
        baseline_reader_before = int(engine.reader.report()["bytes_read"])
        baseline_tokens = [first_token]
        current = first_token
        baseline_step_seconds = 0.0
        baseline_lm_seconds = 0.0
        for _ in range(TRANSITIONS):
            t0 = time.monotonic()
            hidden = engine.step(current)
            baseline_step_seconds += time.monotonic() - t0
            t0 = time.monotonic()
            logits = engine.logits(hidden)
            baseline_lm_seconds += time.monotonic() - t0
            current = _top1(logits)
            baseline_tokens.append(current)
        baseline_elapsed = time.monotonic() - baseline_started
        baseline_k3_bytes = int(engine.reader.report()["bytes_read"]) - baseline_reader_before
        baseline_final = reuse.capture_state(engine)
        baseline_final_digest = reuse.snapshot_digest(baseline_final)

        if baseline_tokens != EXPECTED_TOKENS:
            raise RuntimeError(f"baseline oracle changed: {baseline_tokens}")

        # Restore only target state; MTP has only consumed the common prefix.
        reuse.restore_state(engine, prefix_snap)

        # ---- Transactional MTP-1 speculative path. ----
        spec_started = time.monotonic()
        spec_reader_before = int(engine.reader.report()["bytes_read"])
        spec_tokens = [first_token]
        current = first_token
        prev_h = prefix_h
        rounds: list[dict[str, Any]] = []
        accepted_count = 0
        spec_target_lm_bytes = 0
        spec_mtp_lm_bytes = 0
        spec_target_pair_seconds = 0.0
        spec_target_lm_seconds = 0.0
        spec_mtp_seconds = 0.0
        spec_mtp_lm_seconds = 0.0
        spec_mtp_hit_catchup_seconds = 0.0
        spec_checkpoint_copy_seconds = 0.0
        spec_rollback_seconds = 0.0

        tensor = engine.tensors["output.weight"]
        lm_head_bytes = int(tensor.nbytes)
        if lm_head_bytes != 1_350_860_800:
            raise RuntimeError(f"unexpected LM-head bytes: {lm_head_bytes}")

        while len(spec_tokens) - 1 < TRANSITIONS:
            round_index = len(rounds)
            position = int(engine.position)

            draft = mtp.draft(current, prev_h, position)
            draft_token = int(draft["token"])
            spec_mtp_seconds += float(draft["elapsed_seconds"])
            spec_mtp_lm_seconds += float(draft["lm_head_seconds"])
            spec_mtp_lm_bytes += lm_head_bytes

            hidden_a, hidden_b, txn, pair_seconds, pair_k3_bytes = tx._run_pair_tx(
                engine, current, draft_token)
            spec_target_pair_seconds += pair_seconds
            spec_checkpoint_copy_seconds += float(txn["checkpoint_copy_seconds"])

            t0 = time.monotonic()
            logits_a, logits_b, pair_head_bytes = lm_pair._stream_pair_counted(
                args.model,
                tensor,
                engine.runtime,
                hidden_a,
                hidden_b,
                engine.output_norm_w,
            )
            target_lm_seconds = time.monotonic() - t0
            spec_target_lm_seconds += target_lm_seconds
            spec_target_lm_bytes += int(pair_head_bytes)

            verify_a = _top1(logits_a)
            verify_b = _top1(logits_b)
            accepted = draft_token == verify_a
            accepted_count += int(accepted)
            norm_a = gdn.rms_norm(hidden_a, engine.output_norm_w)

            record: dict[str, Any] = {
                "round": round_index,
                "position": position,
                "input_token": int(current),
                "mtp_draft_token": draft_token,
                "target_verify_a": verify_a,
                "target_verify_b": verify_b,
                "accepted": accepted,
                "pair_k3_bytes": int(pair_k3_bytes),
                "pair_seconds": pair_seconds,
                "target_pair_lm_bytes": int(pair_head_bytes),
                "target_pair_lm_seconds": target_lm_seconds,
                "mtp_draft_seconds": float(draft["elapsed_seconds"]),
                "mtp_lm_head_seconds": float(draft["lm_head_seconds"]),
                "checkpoint_logical_bytes": int(txn["logical_bytes"]),
                "checkpoint_copy_seconds": float(txn["checkpoint_copy_seconds"]),
            }

            if accepted:
                # The target has already processed both A and accepted B.  Emit
                # B and C, then advance the MTP KV history through B without
                # redundantly drafting from B.
                spec_tokens.append(verify_a)
                spec_tokens.append(verify_b)
                t0 = time.monotonic()
                mtp.catchup(draft_token, norm_a, position + 1)
                hit_catchup_seconds = time.monotonic() - t0
                spec_mtp_hit_catchup_seconds += hit_catchup_seconds
                prev_h = gdn.rms_norm(hidden_b, engine.output_norm_w)
                current = verify_b
                record["committed_target_positions"] = 2
                record["rollback_seconds"] = 0.0
                record["mtp_hit_catchup_seconds"] = hit_catchup_seconds
            else:
                # Keep A, discard speculative B exactly, and continue from the
                # true target token after A.  MTP already consumed A, which is
                # precisely the history required by the next round.
                rollback_seconds = tx.rollback_second(engine, txn)
                spec_rollback_seconds += rollback_seconds
                spec_tokens.append(verify_a)
                prev_h = norm_a
                current = verify_a
                record["committed_target_positions"] = 1
                record["rollback_seconds"] = rollback_seconds
                record["mtp_hit_catchup_seconds"] = 0.0

            rounds.append(record)
            if len(rounds) > TRANSITIONS:
                raise RuntimeError("speculative loop failed to make progress")

        spec_elapsed = time.monotonic() - spec_started
        spec_k3_bytes = int(engine.reader.report()["bytes_read"]) - spec_reader_before

        # This pinned gate must end exactly at three transitions, not overshoot.
        if spec_tokens != EXPECTED_TOKENS:
            raise RuntimeError(f"speculative tokens differ from target: {spec_tokens}")
        if [bool(r["accepted"]) for r in rounds] != EXPECTED_BRANCHES:
            raise RuntimeError(f"pinned miss/hit pattern changed: {rounds}")
        state_exact, state_mismatch = reuse.compare_current_to_snapshot(engine, baseline_final)
        if not state_exact:
            raise RuntimeError(f"speculative final target state mismatch: {state_mismatch}")
        spec_final_digest = reuse.snapshot_digest(reuse.capture_state(engine))
        if spec_final_digest != baseline_final_digest:
            raise RuntimeError("speculative final state digest differs from baseline")
        if int(engine.position) != len(mtp.cache["k"]):
            raise RuntimeError(
                f"MTP cache alignment mismatch: target pos={engine.position} mtp={len(mtp.cache['k'])}")

        expected_baseline_k3 = TRANSITIONS * 21_127_430_144
        expected_spec_k3 = len(rounds) * 21_127_430_144
        if baseline_k3_bytes != expected_baseline_k3:
            raise RuntimeError(f"baseline K3 traffic changed: {baseline_k3_bytes}")
        if spec_k3_bytes != expected_spec_k3:
            raise RuntimeError(f"speculative K3 traffic changed: {spec_k3_bytes}")
        if not bool(engine.reader.report().get("direct_io")):
            raise RuntimeError("end-to-end speculative gate requires direct I/O")

        baseline_target_lm_bytes = TRANSITIONS * lm_head_bytes
        spec_total_lm_bytes = spec_target_lm_bytes + spec_mtp_lm_bytes
        payload = {
            "schema": "qwen38-mtp1-speculative-e2e-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prefix_token": PREFIX_TOKEN,
            "target_tokens": baseline_tokens,
            "speculative_tokens": spec_tokens,
            "target_sequence_exact": spec_tokens == baseline_tokens,
            "final_target_state_bitwise_exact": state_exact,
            "final_target_state_mismatch": state_mismatch,
            "baseline_final_state_sha256": baseline_final_digest,
            "speculative_final_state_sha256": spec_final_digest,
            "baseline": {
                "target_steps": TRANSITIONS,
                "elapsed_seconds": baseline_elapsed,
                "target_step_seconds": baseline_step_seconds,
                "target_lm_head_seconds": baseline_lm_seconds,
                "k3_bytes": baseline_k3_bytes,
                "target_lm_head_bytes": baseline_target_lm_bytes,
                "decoder_plus_target_head_bytes": baseline_k3_bytes + baseline_target_lm_bytes,
            },
            "speculative": {
                "rounds": len(rounds),
                "accepted_count": accepted_count,
                "acceptance_rate": accepted_count / len(rounds),
                "elapsed_seconds": spec_elapsed,
                "target_pair_seconds": spec_target_pair_seconds,
                "target_pair_lm_seconds": spec_target_lm_seconds,
                "mtp_seconds": spec_mtp_seconds,
                "mtp_lm_head_seconds": spec_mtp_lm_seconds,
                "mtp_hit_catchup_seconds": spec_mtp_hit_catchup_seconds,
                "checkpoint_copy_seconds": spec_checkpoint_copy_seconds,
                "rollback_seconds": spec_rollback_seconds,
                "k3_bytes": spec_k3_bytes,
                "effective_k3_bytes_per_transition": spec_k3_bytes / TRANSITIONS,
                "target_pair_lm_bytes": spec_target_lm_bytes,
                "mtp_lm_head_bytes": spec_mtp_lm_bytes,
                "total_lm_head_bytes": spec_total_lm_bytes,
                "decoder_plus_all_lm_head_bytes": spec_k3_bytes + spec_total_lm_bytes,
                "mtp_cache_positions": len(mtp.cache["k"]),
                "target_position": int(engine.position),
                "round_records": rounds,
            },
            "comparison": {
                "elapsed_speedup": baseline_elapsed / spec_elapsed,
                "k3_bytes_saved": baseline_k3_bytes - spec_k3_bytes,
                "k3_reduction_fraction": 1.0 - spec_k3_bytes / baseline_k3_bytes,
                "decoder_plus_lm_bytes_saved":
                    (baseline_k3_bytes + baseline_target_lm_bytes)
                    - (spec_k3_bytes + spec_total_lm_bytes),
                "decoder_plus_lm_reduction_fraction": 1.0
                    - (spec_k3_bytes + spec_total_lm_bytes)
                    / (baseline_k3_bytes + baseline_target_lm_bytes),
            },
            "shared_prefix": {
                "initial_target_lm_head_seconds": initial_lm_seconds,
                "mtp_catchup_seconds": prefix_mtp_catchup_seconds,
            },
            "mtp": mtp.report(),
            "target_reader": engine.reader.report(),
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "tokens": spec_tokens,
            "branches": [int(x) for x in EXPECTED_BRANCHES],
            "baseline_seconds": baseline_elapsed,
            "speculative_seconds": spec_elapsed,
            "speedup": payload["comparison"]["elapsed_speedup"],
            "baseline_k3_bytes": baseline_k3_bytes,
            "speculative_k3_bytes": spec_k3_bytes,
            "effective_k3_bytes_per_transition": payload["speculative"]["effective_k3_bytes_per_transition"],
            "baseline_target_lm_bytes": baseline_target_lm_bytes,
            "speculative_total_lm_bytes": spec_total_lm_bytes,
            "decoder_plus_lm_reduction_fraction": payload["comparison"]["decoder_plus_lm_reduction_fraction"],
            "final_target_state_bitwise_exact": state_exact,
            "mtp_cache_positions": len(mtp.cache["k"]),
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_MTP1_SPECULATIVE_E2E_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert EXPECTED_TOKENS == [11, 353, 2688, 264]
    assert EXPECTED_BRANCHES == [False, True]
    assert TRANSITIONS == 3
    print(json.dumps({
        "schema": "qwen38-mtp1-speculative-e2e-sanity-v1",
        "status": "PASS",
        "transitions": TRANSITIONS,
        "expected_target_streams": 2,
        "branches": ["miss", "hit"],
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
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
