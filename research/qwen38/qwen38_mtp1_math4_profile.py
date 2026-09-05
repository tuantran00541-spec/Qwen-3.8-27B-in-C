#!/usr/bin/env python3
"""Stage-timed wrapper for the compact staged-prefill MTP-1 math benchmark.

This intentionally does not change target arithmetic or scheduling.  It wraps
existing benchmark/runtime entry points with flushed progress markers so a CI
timeout still identifies the stage that consumed the wall clock.
"""
from __future__ import annotations

import json
import time
from typing import Any

import qwen38_mtp1_math4_benchmark as bench

bench.MATH4_PROMPT = """4 final answers only.
1 sqrt(x+6)+sqrt(x-3)=5,x>=3
2 5R4B3G urn,3 draws no replacement:P(exactly 2 colors)
3 7^2026 mod1000
4 positive a<=b,1/a+1/b=1/6:all pairs"""

_RUN_STARTED = time.monotonic()
_PROMPT_TOKENS = 0


def emit(stage: str, event: str, **fields: Any) -> None:
    payload = {
        "marker": "QWEN38_MATH4_STAGE",
        "stage": stage,
        "event": event,
        "elapsed_from_wrapper_start_seconds": time.monotonic() - _RUN_STARTED,
    }
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


# Tokenizer / prompt envelope.
_orig_load_tokenizer = bench.gen.load_tokenizer


def _load_tokenizer(*args, **kwargs):
    emit("tokenizer_load", "begin")
    t0 = time.monotonic()
    out = _orig_load_tokenizer(*args, **kwargs)
    emit("tokenizer_load", "end", seconds=time.monotonic() - t0)
    return out


bench.gen.load_tokenizer = _load_tokenizer
_orig_encode_prompt = bench.gen.encode_prompt


def _encode_prompt(*args, **kwargs):
    global _PROMPT_TOKENS
    t0 = time.monotonic()
    rendered, ids = _orig_encode_prompt(*args, **kwargs)
    _PROMPT_TOKENS = len(ids)
    emit("prompt_encode", "end", seconds=time.monotonic() - t0, prompt_tokens=_PROMPT_TOKENS)
    return rendered, ids


bench.gen.encode_prompt = _encode_prompt


# Target engine initialization.
_Engine = bench.gen.StatefulK3Generator


def _instrumented_engine(*args, **kwargs):
    emit("target_engine_init", "begin")
    t0 = time.monotonic()
    out = _Engine(*args, **kwargs)
    emit("target_engine_init", "end", seconds=time.monotonic() - t0)
    return out


bench.gen.StatefulK3Generator = _instrumented_engine


# MTP resident block initialization.
_orig_mtp_init = bench.MTPBlock.__init__


def _mtp_init(self, *args, **kwargs):
    emit("mtp_init", "begin")
    t0 = time.monotonic()
    _orig_mtp_init(self, *args, **kwargs)
    emit(
        "mtp_init",
        "end",
        seconds=time.monotonic() - t0,
        resident_weight_bytes=int(self.resident_weight_bytes),
    )


bench.MTPBlock.__init__ = _mtp_init


# Staged target prompt prefill.  The begin marker is especially important: if
# the process times out inside the monolithic full64 pass, the artifact still
# proves that prefill was the active stage.
_orig_step_block_many = bench.prefill_many.step_block_many


def _step_block_many(engine, token_ids):
    emit("target_staged_prefill", "begin", prompt_tokens=len(token_ids))
    t0 = time.monotonic()
    out = _orig_step_block_many(engine, token_ids)
    emit(
        "target_staged_prefill",
        "end",
        seconds=time.monotonic() - t0,
        prompt_tokens=len(token_ids),
        target_position=int(engine.position),
        reader_bytes=int(engine.reader.report()["bytes_read"]),
    )
    return out


bench.prefill_many.step_block_many = _step_block_many


# MTP prompt catch-up and later hit/single-tail catch-up use the same method.
_orig_catchup = bench.MTPBlock.catchup


def _catchup(self, token_id, h_prev, position):
    phase = "prompt_mtp_catchup" if int(position) < _PROMPT_TOKENS else "generation_mtp_catchup"
    if phase != "prompt_mtp_catchup" or int(position) == 0 or (int(position) + 1) % 8 == 0:
        emit(phase, "position_begin", position=int(position), prompt_tokens=_PROMPT_TOKENS)
    t0 = time.monotonic()
    out = _orig_catchup(self, token_id, h_prev, position)
    seconds = time.monotonic() - t0
    if phase != "prompt_mtp_catchup" or int(position) == 0 or (int(position) + 1) % 8 == 0 or int(position) + 1 == _PROMPT_TOKENS:
        emit(
            phase,
            "position_end",
            position=int(position),
            seconds=seconds,
            cache_positions=len(self.cache["k"]),
            prompt_tokens=_PROMPT_TOKENS,
        )
    return out


bench.MTPBlock.catchup = _catchup


# LM-head and speculative-generation stages.
_orig_one = bench.lm_pair._stream_one_counted


def _stream_one(*args, **kwargs):
    emit("target_lm_head_single", "begin")
    t0 = time.monotonic()
    out = _orig_one(*args, **kwargs)
    emit("target_lm_head_single", "end", seconds=time.monotonic() - t0, bytes=int(out[1]))
    return out


bench.lm_pair._stream_one_counted = _stream_one
_orig_pair_head = bench.lm_pair._stream_pair_counted


def _stream_pair(*args, **kwargs):
    emit("target_lm_head_pair", "begin")
    t0 = time.monotonic()
    out = _orig_pair_head(*args, **kwargs)
    emit("target_lm_head_pair", "end", seconds=time.monotonic() - t0, bytes=int(out[2]))
    return out


bench.lm_pair._stream_pair_counted = _stream_pair
_orig_pair_tx = bench.tx._run_pair_tx


def _run_pair_tx(engine, token_a, token_b):
    emit(
        "target_transactional_pair",
        "begin",
        position=int(engine.position),
        token_a=int(token_a),
        token_b=int(token_b),
    )
    t0 = time.monotonic()
    out = _orig_pair_tx(engine, token_a, token_b)
    emit(
        "target_transactional_pair",
        "end",
        seconds=time.monotonic() - t0,
        pair_reported_seconds=float(out[3]),
        pair_k3_bytes=int(out[4]),
        target_position=int(engine.position),
    )
    return out


bench.tx._run_pair_tx = _run_pair_tx
_orig_draft = bench.MTPBlock.draft


def _draft(self, token_id, h_prev, position):
    emit("mtp_draft", "begin", position=int(position), token_id=int(token_id))
    t0 = time.monotonic()
    out = _orig_draft(self, token_id, h_prev, position)
    emit(
        "mtp_draft",
        "end",
        seconds=time.monotonic() - t0,
        reported_seconds=float(out["elapsed_seconds"]),
        lm_head_seconds=float(out["lm_head_seconds"]),
        draft_token=int(out["token"]),
    )
    return out


bench.MTPBlock.draft = _draft


if __name__ == "__main__":
    emit("wrapper", "start")
    raise SystemExit(bench.main())
