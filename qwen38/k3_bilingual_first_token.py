#!/usr/bin/env python3
"""Real VI/EN first-token generation smoke over the proven K3 Qwen3.8 path."""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from huggingface_hub import hf_hub_download

from core import LAYERS as TOTAL_LAYERS, MODEL_ID, PINNED_REVISION
from k3_full_attention_gate import causal_mask
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
from k3_stream import needed_shards, pack_layers
from k3_windowed_chain_gate import remove_file
from k3_full_cached_logits_gate import process_window

PROMPTS = {
    "vi": "Hãy trả lời đúng một từ tiếng Việt, không giải thích: Hai cộng hai bằng mấy?",
    "en": "Answer with exactly one English word and no explanation: Two plus two equals what?",
}
EXPECTED = {"vi": "bốn", "en": "four"}
UPSTREAM_GATE = {
    "schema": "qwen38-k3-full-cached-logits-gate-v1",
    "run_id": 33361238941,
    "artifact_id": 9746977220,
    "sha256": "ac52068acf13203a8ffcb9a07dcc662121280878a36e4b4cda1832556053f5d7",
    "status": "PASS",
}


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def render(tokenizer, prompt: str) -> tuple[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    return rendered, encoded["input_ids"].to(torch.long)


def pack_decoder_windows(
    model_dir: Path,
    trunk_dir: Path,
    weight_map: dict[str, str],
    window_size: int,
    protected_shards: set[str],
) -> tuple[list[dict], int]:
    specs: list[dict] = []
    retained = 0
    for start in range(0, TOTAL_LAYERS, window_size):
        stop = start + window_size
        layers = tuple(range(start, stop))
        shards = needed_shards(weight_map, layers)
        for shard in shards:
            hf_hub_download(
                MODEL_ID, filename=shard, revision=PINNED_REVISION, local_dir=str(model_dir)
            )
        out_bin = trunk_dir / f"layers{start}-{stop-1}.trunk.bin"
        out_idx = trunk_dir / f"layers{start}-{stop-1}.trunk.json"
        manifest = pack_layers(
            model_dir, weight_map, layers, out_bin, out_idx,
            model_id=MODEL_ID, revision=PINNED_REVISION,
        )
        meta = {int(item["layer"]): item for item in manifest["layers"]}
        specs.append({
            "layers": list(layers),
            "out_bin": str(out_bin),
            "out_idx": str(out_idx),
            "meta_by_layer": meta,
            "max_layer_bytes": max(int(item["read_bytes"]) for item in manifest["layers"]),
            "expected_bytes": sum(int(meta[layer]["read_bytes"]) for layer in layers),
            "packed_file_bytes": int(manifest["packed_file_bytes"]),
        })
        retained += int(manifest["packed_file_bytes"])
        for shard in shards:
            if shard not in protected_shards:
                remove_file(model_dir / shard)
        gc.collect()
        print(f"QWEN38_REAL_PACK layers={start}-{stop-1}", flush=True)
    return specs, retained


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
) -> dict:
    started = time.monotonic()
    rendered, token_ids = render(tokenizer, prompt)
    reference = reference_embedding(model_dir, weight_map, token_ids)
    candidate, embed_runtime = streamed_embedding(embed_bin, embed_idx, embed_meta, token_ids)
    embed_metrics = metrics(reference, candidate)
    if not embed_metrics["exact_equal"]:
        raise RuntimeError(f"{lang}: real prompt embedding mismatch: {embed_metrics}")

    seq_len = int(token_ids.shape[-1])
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    with torch.inference_mode():
        position_embeddings = rotary(reference, position_ids)
    full_mask = causal_mask(seq_len)
    reference_cache = DynamicCache(config=cfg)
    linear_cache: dict[int, dict] = {}
    full_cache: dict[int, dict] = {}
    windows = []
    cache_items = []
    stream_bytes = int(embed_runtime["bytes_read"])

    for spec in specs:
        reference, candidate, win, caches = process_window(
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
        windows.append(win)
        cache_items.extend(caches)
        stream_bytes += int(win["runtime"]["bytes_read"])

    decoder_metrics = metrics(reference, candidate)
    ref_norm, ref_logits = reference_tail(reference, cfg, model_dir, weight_map)
    cand_norm, cand_logits, tail_runtime = streamed_tail(candidate, cfg, tail_bin, tail_idx, tail_meta)
    norm_metrics = metrics(ref_norm, cand_norm)
    logits_metrics = metrics(ref_logits, cand_logits)
    stream_bytes += int(tail_runtime["bytes_read"])
    if not all((decoder_metrics["exact_equal"], norm_metrics["exact_equal"], logits_metrics["exact_equal"])):
        raise RuntimeError(f"{lang}: real prompt equivalence failed")

    ref_id = int(torch.argmax(ref_logits[0, -1]).item())
    cand_id = int(torch.argmax(cand_logits[0, -1]).item())
    if ref_id != cand_id:
        raise RuntimeError(f"{lang}: greedy token mismatch {ref_id}!={cand_id}")
    token_text = tokenizer.decode([cand_id], skip_special_tokens=True).strip()
    expected = EXPECTED[lang]
    semantic_match = norm_text(token_text) == norm_text(expected)
    print(
        f"QWEN38_REAL_REPLY lang={lang} token_id={cand_id} "
        f"semantic_match={semantic_match} text={json.dumps(token_text, ensure_ascii=False)}",
        flush=True,
    )
    return {
        "prompt": prompt,
        "rendered_prompt": rendered,
        "prompt_tokens": seq_len,
        "first_token_id": cand_id,
        "first_token_text": token_text,
        "expected": expected,
        "semantic_match": semantic_match,
        "embedding_metrics": embed_metrics,
        "decoder_metrics": decoder_metrics,
        "final_norm_metrics": norm_metrics,
        "logits_metrics": logits_metrics,
        "windows_exact": all(
            item["hidden_metrics"]["exact_equal"] and item["cache_exact"] for item in windows
        ),
        "cache_items": len(cache_items),
        "cache_exact": all(item["cache_exact"] for item in windows),
        "reference_cache_seq_len": int(reference_cache.get_seq_length()),
        "stream_bytes": stream_bytes,
        "seconds": time.monotonic() - started,
        "max_rss_gib": max_rss_gib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.window_size < 2 or TOTAL_LAYERS % args.window_size:
        raise SystemExit("--window-size must divide 64 and be >=2")

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
    preview = {lang: render(tokenizer, prompt)[1].shape[-1] for lang, prompt in PROMPTS.items()}
    tokenizer_seconds = time.monotonic() - tokenizer_started
    print(f"QWEN38_REAL_TOKENIZER prompt_tokens={preview}", flush=True)

    config, index, validated = download_metadata(model_dir)
    text_config = config["text_config"]
    cfg = Qwen3_5TextConfig(**text_config)
    cfg._attn_implementation = "eager"
    weight_map = index["weight_map"]
    layout = validate_global_layout(weight_map, text_config)

    embed_shards = set(download_names(model_dir, weight_map, [EMBED_NAME]))
    embed_bin = trunk_dir / "embedding.trunk.bin"
    embed_idx = trunk_dir / "embedding.trunk.json"
    embed_manifest = pack_named_group(model_dir, weight_map, [EMBED_NAME], -1, embed_bin, embed_idx)
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
        )

    exact = all(
        r["embedding_metrics"]["exact_equal"]
        and r["decoder_metrics"]["exact_equal"]
        and r["final_norm_metrics"]["exact_equal"]
        and r["logits_metrics"]["exact_equal"]
        and r["windows_exact"]
        and r["cache_exact"]
        for r in replies.values()
    )
    semantic_all = all(r["semantic_match"] for r in replies.values())
    result = {
        "schema": "qwen38-k3-bilingual-first-token-v1",
        "status": "PASS" if exact else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "global_layout": layout,
        "upstream_equivalence": UPSTREAM_GATE,
        "tokenizer_seconds": tokenizer_seconds,
        "thinking": False,
        "generation_mode": "real chat-template greedy first token",
        "replies": replies,
        "semantic_smoke_all_match": semantic_all,
        "retained_decoder_trunk_bytes": retained_decoder_bytes,
        "retained_embedding_trunk_bytes": int(embed_manifest["packed_file_bytes"]),
        "retained_tail_trunk_bytes": int(tail_manifest["packed_file_bytes"]),
        "total_stream_bytes": sum(r["stream_bytes"] for r in replies.values()),
        "min_disk_free_bytes": min_disk_free,
        "max_rss_gib": max_rss_gib(),
        "total_seconds": time.monotonic() - started,
        "generation_attempted": True,
        "generation_completed": exact,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    for shard in embed_shards | tail_shards:
        remove_file(model_dir / shard)
    remove_file(embed_bin); remove_file(embed_idx)
    remove_file(tail_bin); remove_file(tail_idx)
    for spec in specs:
        remove_file(Path(spec["out_bin"])); remove_file(Path(spec["out_idx"]))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
