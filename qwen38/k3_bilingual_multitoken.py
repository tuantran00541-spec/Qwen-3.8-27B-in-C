#!/usr/bin/env python3
"""Real VI/EN multi-token greedy generation over the proven K3 Qwen3.8 path.

This extends the first-token smoke with cached one-token decode. It stops early
when the decoded answer exactly matches the expected one-word answer, so English
normally ends after one token while Vietnamese can prove tokenizer continuation
(e.g. B + ... -> Bốn) without unnecessary full-model passes.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from core import MODEL_ID, PINNED_REVISION
from k3_bilingual_first_token import (
    EXPECTED,
    PROMPTS,
    UPSTREAM_GATE,
    max_rss_gib,
    norm_text,
    pack_decoder_windows,
    render,
)
from k3_full_cached_logits_gate import process_window
from k3_linear_layer_gate import download_metadata, metrics
from k3_logits_gate import (
    EMBED_NAME,
    FINAL_NORM_NAME,
    LM_HEAD_NAME,
    download_names,
    pack_named_group,
    reference_embedding,
    reference_tail,
    streamed_embedding,
    streamed_tail,
    validate_global_layout,
)
from k3_windowed_chain_gate import remove_file


def exact_tail(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    cfg: Qwen3_5TextConfig,
    model_dir: Path,
    weight_map: dict[str, str],
    tail_bin: Path,
    tail_idx: Path,
    tail_meta: dict,
) -> tuple[int, dict, dict, int]:
    ref_norm, ref_logits = reference_tail(reference, cfg, model_dir, weight_map)
    cand_norm, cand_logits, tail_runtime = streamed_tail(candidate, cfg, tail_bin, tail_idx, tail_meta)
    norm_metrics = metrics(ref_norm, cand_norm)
    logits_metrics = metrics(ref_logits, cand_logits)
    if not norm_metrics["exact_equal"] or not logits_metrics["exact_equal"]:
        raise RuntimeError(
            f"tail mismatch norm={norm_metrics} logits={logits_metrics}"
        )
    ref_id = int(torch.argmax(ref_logits[0, -1]).item())
    cand_id = int(torch.argmax(cand_logits[0, -1]).item())
    if ref_id != cand_id:
        raise RuntimeError(f"greedy token mismatch {ref_id}!={cand_id}")
    return cand_id, norm_metrics, logits_metrics, int(tail_runtime["bytes_read"])


def run_prompt(
    *,
    lang: str,
    prompt: str,
    tokenizer,
    cfg: Qwen3_5TextConfig,
    text_config: dict,
    weight_map: dict[str, str],
    model_dir: Path,
    embed_bin: Path,
    embed_idx: Path,
    embed_meta: dict,
    specs: list[dict],
    tail_bin: Path,
    tail_idx: Path,
    tail_meta: dict,
    max_new_tokens: int,
) -> dict:
    started = time.monotonic()
    rendered, token_ids = render(tokenizer, prompt)
    seq_len = int(token_ids.shape[-1])

    reference = reference_embedding(model_dir, weight_map, token_ids)
    candidate, embed_runtime = streamed_embedding(embed_bin, embed_idx, embed_meta, token_ids)
    embed_metrics = metrics(reference, candidate)
    if not embed_metrics["exact_equal"]:
        raise RuntimeError(f"{lang}: prompt embedding mismatch: {embed_metrics}")

    reference_cache = DynamicCache(config=cfg)
    linear_cache: dict[int, dict] = {}
    full_cache: dict[int, dict] = {}
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        position_embeddings = rotary(reference, position_ids)
    full_mask = torch.triu(
        torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )

    stream_bytes = int(embed_runtime["bytes_read"])
    prefill_windows = []
    for spec in specs:
        reference, candidate, win, _ = process_window(
            phase="prefill",
            spec=spec,
            reference=reference,
            candidate=candidate,
            cfg=cfg,
            text_config=text_config,
            weight_map=weight_map,
            reference_cache=reference_cache,
            linear_cache=linear_cache,
            full_cache=full_cache,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            full_mask=full_mask,
        )
        prefill_windows.append(win)
        stream_bytes += int(win["runtime"]["bytes_read"])
    prefill_metrics = metrics(reference, candidate)
    if not prefill_metrics["exact_equal"] or not all(
        w["hidden_metrics"]["exact_equal"] and w["cache_exact"] for w in prefill_windows
    ):
        raise RuntimeError(f"{lang}: prefill/cache mismatch")

    next_id, norm_metrics, logits_metrics, tail_bytes = exact_tail(
        reference,
        candidate,
        cfg=cfg,
        model_dir=model_dir,
        weight_map=weight_map,
        tail_bin=tail_bin,
        tail_idx=tail_idx,
        tail_meta=tail_meta,
    )
    stream_bytes += tail_bytes
    generated = [next_id]
    steps = [{
        "step": 0,
        "phase": "prefill_logits",
        "token_id": next_id,
        "token_piece": tokenizer.decode([next_id], skip_special_tokens=True),
        "decoder_metrics": prefill_metrics,
        "final_norm_metrics": norm_metrics,
        "logits_metrics": logits_metrics,
        "cache_seq_len": int(reference_cache.get_seq_length()),
    }]

    expected = EXPECTED[lang]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    semantic_match = norm_text(decoded) == norm_text(expected)
    eos_id = tokenizer.eos_token_id

    while len(generated) < max_new_tokens and not semantic_match:
        if eos_id is not None and generated[-1] == int(eos_id):
            break
        # Feed the previously predicted token through the cached decoder to
        # produce logits for the following token.
        decode_ids = torch.tensor([[generated[-1]]], dtype=torch.long)
        reference = reference_embedding(model_dir, weight_map, decode_ids)
        candidate, decode_embed_runtime = streamed_embedding(
            embed_bin, embed_idx, embed_meta, decode_ids
        )
        decode_embed_metrics = metrics(reference, candidate)
        if not decode_embed_metrics["exact_equal"]:
            raise RuntimeError(f"{lang}: decode embedding mismatch")
        stream_bytes += int(decode_embed_runtime["bytes_read"])

        position = seq_len + len(generated) - 1
        decode_pos_ids = torch.tensor([[position]], dtype=torch.long)
        with torch.inference_mode():
            decode_pos = rotary(reference, decode_pos_ids)
        # The current token is appended to a cache containing `position`
        # previous tokens, so it may attend to all position+1 keys.
        decode_mask = torch.zeros((1, 1, 1, position + 1), dtype=torch.float32)

        decode_windows = []
        for spec in specs:
            reference, candidate, win, _ = process_window(
                phase="decode",
                spec=spec,
                reference=reference,
                candidate=candidate,
                cfg=cfg,
                text_config=text_config,
                weight_map=weight_map,
                reference_cache=reference_cache,
                linear_cache=linear_cache,
                full_cache=full_cache,
                position_embeddings=decode_pos,
                position_ids=decode_pos_ids,
                full_mask=decode_mask,
            )
            decode_windows.append(win)
            stream_bytes += int(win["runtime"]["bytes_read"])
        decode_metrics = metrics(reference, candidate)
        if not decode_metrics["exact_equal"] or not all(
            w["hidden_metrics"]["exact_equal"] and w["cache_exact"] for w in decode_windows
        ):
            raise RuntimeError(f"{lang}: cached decode mismatch at step {len(generated)}")

        next_id, step_norm, step_logits, tail_bytes = exact_tail(
            reference,
            candidate,
            cfg=cfg,
            model_dir=model_dir,
            weight_map=weight_map,
            tail_bin=tail_bin,
            tail_idx=tail_idx,
            tail_meta=tail_meta,
        )
        stream_bytes += tail_bytes
        generated.append(next_id)
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        semantic_match = norm_text(decoded) == norm_text(expected)
        steps.append({
            "step": len(generated) - 1,
            "phase": "cached_decode_logits",
            "token_id": next_id,
            "token_piece": tokenizer.decode([next_id], skip_special_tokens=True),
            "decoded_so_far": decoded,
            "embedding_metrics": decode_embed_metrics,
            "decoder_metrics": decode_metrics,
            "final_norm_metrics": step_norm,
            "logits_metrics": step_logits,
            "cache_seq_len": int(reference_cache.get_seq_length()),
        })
        print(
            f"QWEN38_MULTI_STEP lang={lang} n={len(generated)} token_id={next_id} "
            f"semantic_match={semantic_match} text={json.dumps(decoded, ensure_ascii=False)}",
            flush=True,
        )

    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    semantic_match = norm_text(decoded) == norm_text(expected)
    print(
        f"QWEN38_MULTI_REPLY lang={lang} tokens={generated} "
        f"semantic_match={semantic_match} text={json.dumps(decoded, ensure_ascii=False)}",
        flush=True,
    )
    return {
        "prompt": prompt,
        "rendered_prompt": rendered,
        "prompt_tokens": seq_len,
        "generated_token_ids": generated,
        "generated_tokens": len(generated),
        "text": decoded,
        "expected": expected,
        "semantic_match": semantic_match,
        "steps": steps,
        "reference_cache_seq_len": int(reference_cache.get_seq_length()),
        "stream_bytes": stream_bytes,
        "seconds": time.monotonic() - started,
        "max_rss_gib": max_rss_gib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >=1")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    min_disk_free = shutil.disk_usage(root).free

    tokenizer_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=PINNED_REVISION, use_fast=True
    )
    tokenizer_seconds = time.monotonic() - tokenizer_started

    config, index, validated = download_metadata(model_dir)
    text_config = config["text_config"]
    cfg = Qwen3_5TextConfig(**text_config)
    cfg._attn_implementation = "eager"
    weight_map = index["weight_map"]
    layout = validate_global_layout(weight_map, text_config)

    embed_shards = set(download_names(model_dir, weight_map, [EMBED_NAME]))
    embed_bin = trunk_dir / "embedding.trunk.bin"
    embed_idx = trunk_dir / "embedding.trunk.json"
    embed_manifest = pack_named_group(
        model_dir, weight_map, [EMBED_NAME], -1, embed_bin, embed_idx
    )
    embed_meta = embed_manifest["layers"][0]

    specs, retained_decoder_bytes = pack_decoder_windows(
        model_dir, trunk_dir, weight_map, args.window_size, embed_shards
    )
    min_disk_free = min(min_disk_free, shutil.disk_usage(root).free)

    tail_shards = set(download_names(model_dir, weight_map, [FINAL_NORM_NAME, LM_HEAD_NAME]))
    tail_bin = trunk_dir / "tail.trunk.bin"
    tail_idx = trunk_dir / "tail.trunk.json"
    tail_manifest = pack_named_group(
        model_dir, weight_map, [FINAL_NORM_NAME, LM_HEAD_NAME], 64, tail_bin, tail_idx
    )
    tail_meta = tail_manifest["layers"][0]
    min_disk_free = min(min_disk_free, shutil.disk_usage(root).free)

    replies = {}
    for lang in ("vi", "en"):
        replies[lang] = run_prompt(
            lang=lang,
            prompt=PROMPTS[lang],
            tokenizer=tokenizer,
            cfg=cfg,
            text_config=text_config,
            weight_map=weight_map,
            model_dir=model_dir,
            embed_bin=embed_bin,
            embed_idx=embed_idx,
            embed_meta=embed_meta,
            specs=specs,
            tail_bin=tail_bin,
            tail_idx=tail_idx,
            tail_meta=tail_meta,
            max_new_tokens=args.max_new_tokens,
        )

    semantic_all = all(item["semantic_match"] for item in replies.values())
    result = {
        "schema": "qwen38-k3-bilingual-multitoken-v1",
        "status": "PASS" if semantic_all else "PARTIAL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "global_layout": layout,
        "upstream_equivalence": UPSTREAM_GATE,
        "thinking": False,
        "generation_mode": "real chat-template cached greedy multi-token",
        "max_new_tokens": args.max_new_tokens,
        "replies": replies,
        "semantic_smoke_all_match": semantic_all,
        "retained_decoder_trunk_bytes": retained_decoder_bytes,
        "retained_embedding_trunk_bytes": int(embed_manifest["packed_file_bytes"]),
        "retained_tail_trunk_bytes": int(tail_manifest["packed_file_bytes"]),
        "total_stream_bytes": sum(item["stream_bytes"] for item in replies.values()),
        "min_disk_free_bytes": min_disk_free,
        "max_rss_gib": max_rss_gib(),
        "tokenizer_seconds": tokenizer_seconds,
        "total_seconds": time.monotonic() - started,
        "generation_attempted": True,
        "generation_completed": True,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    for shard in embed_shards | tail_shards:
        remove_file(model_dir / shard)
    remove_file(embed_bin); remove_file(embed_idx)
    remove_file(tail_bin); remove_file(tail_idx)
    for spec in specs:
        remove_file(Path(spec["out_bin"])); remove_file(Path(spec["out_idx"]))

    if not semantic_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
