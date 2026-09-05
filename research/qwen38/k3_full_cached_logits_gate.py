#!/usr/bin/env python3
"""Full 64-layer cached-logits equivalence gate for the K3-style Qwen3.8 runtime.

This is the final composition gate before real prompt generation. It verifies:
1) official token embedding,
2) cached prefill through all 64 decoder layers,
3) one-token cached decode through all 64 decoder layers,
4) every linear-attention conv/recurrent cache and every full-attention KV cache,
5) final RMSNorm and last-token LM-head logits for both prefill and decode.

Decoder weights are downloaded once window-by-window, packed into retained K3
trunks, and source shards are deleted immediately. The independent reference
equations are the official Transformers Qwen3_5DecoderLayer modules, loaded from
the already bit-exact K3 tensor views. Earlier gates independently established
K3-vs-official weight/decoder/logits equality; this gate focuses on cache and
full-path composition without downloading the 54 GB checkpoint a second time.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import time
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
from huggingface_hub import hf_hub_download
from transformers import DynamicCache, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

from core import LAYERS as TOTAL_LAYERS, MODEL_ID, PINNED_REVISION
from k3_cache_gate import (
    candidate_full_cached,
    candidate_linear_cached,
    state_metrics,
    tensor_bytes,
)
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
from k3_stream import K3Trunk, needed_shards, pack_layers
from k3_windowed_chain_gate import remove_file, streamed_state


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def cache_bytes(linear_cache: dict[int, dict], full_cache: dict[int, dict]) -> dict[str, int]:
    return {
        "linear_conv": sum(tensor_bytes(item["conv"]) for item in linear_cache.values()),
        "linear_recurrent": sum(tensor_bytes(item["recurrent"]) for item in linear_cache.values()),
        "full_key": sum(tensor_bytes(item["key"]) for item in full_cache.values()),
        "full_value": sum(tensor_bytes(item["value"]) for item in full_cache.values()),
    }


def candidate_full_lengths(full_cache: dict[int, dict]) -> list[int]:
    return sorted({int(item["key"].shape[-2]) for item in full_cache.values()})


def reference_layer_from_streamed(
    hidden: torch.Tensor,
    cfg: Qwen3_5TextConfig,
    layer: int,
    state: dict[str, torch.Tensor],
    cache: DynamicCache,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    full_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the official decoder module after loading the exact streamed state."""
    module = Qwen3_5DecoderLayer(cfg, layer).to(dtype=torch.bfloat16).eval()
    expected = set(module.state_dict().keys())
    if set(state) != expected:
        raise RuntimeError(
            f"layer {layer} reference state mismatch "
            f"missing={sorted(expected-set(state))} unexpected={sorted(set(state)-expected)}"
        )
    module.load_state_dict(state, strict=True)
    layer_type = cfg.layer_types[layer]
    dummy = torch.empty(0, dtype=torch.float32)
    with torch.inference_mode():
        if layer_type == "linear_attention":
            out = module(
                hidden,
                position_embeddings=(dummy, dummy),
                attention_mask=None,
                position_ids=None,
                past_key_values=cache,
            )
        elif layer_type == "full_attention":
            out = module(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=full_mask,
                position_ids=position_ids,
                past_key_values=cache,
            )
        else:
            raise RuntimeError(f"unsupported layer type {layer_type!r}")
    del module
    gc.collect()
    return out


def compare_layer_cache(
    reference_cache: DynamicCache,
    linear_cache: dict[int, dict],
    full_cache: dict[int, dict],
    layer: int,
    layer_type: str,
) -> dict:
    ref = reference_cache.layers[layer]
    if layer_type == "linear_attention":
        if layer not in linear_cache:
            raise RuntimeError(f"candidate linear cache missing layer {layer}")
        item = linear_cache[layer]
        if not reference_cache.has_previous_state(layer, state_idx=0):
            raise RuntimeError(f"reference linear cache layer {layer} has no previous state")
        return {
            "layer": layer,
            "type": layer_type,
            "conv": state_metrics(ref.conv_states[0], item["conv"]),
            "recurrent": state_metrics(ref.recurrent_states[0], item["recurrent"]),
        }
    if layer_type == "full_attention":
        if layer not in full_cache:
            raise RuntimeError(f"candidate full cache missing layer {layer}")
        item = full_cache[layer]
        return {
            "layer": layer,
            "type": layer_type,
            "key": state_metrics(ref.keys, item["key"]),
            "value": state_metrics(ref.values, item["value"]),
        }
    raise RuntimeError(f"unsupported layer type {layer_type!r}")


def cache_metrics_exact(items: list[dict]) -> bool:
    for item in items:
        for name, value in item.items():
            if name in {"layer", "type"}:
                continue
            if not value["exact_equal"]:
                return False
    return True


def process_window(
    *,
    phase: str,
    spec: dict,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    cfg: Qwen3_5TextConfig,
    text_config: dict,
    weight_map: dict[str, str],
    reference_cache: DynamicCache,
    linear_cache: dict[int, dict],
    full_cache: dict[int, dict],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    full_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict, list[dict]]:
    layers = tuple(spec["layers"])
    meta_by_layer = spec["meta_by_layer"]
    out_bin = Path(spec["out_bin"])
    out_idx = Path(spec["out_idx"])
    max_layer_bytes = int(spec["max_layer_bytes"])
    expected_bytes = int(spec["expected_bytes"])
    started = time.monotonic()
    prefetch_expected = 0
    prefetch_accepted = 0
    layer_cache_metrics: list[dict] = []

    with K3Trunk(
        out_bin,
        out_idx,
        budget_bytes=2 * max_layer_bytes,
        want_ring=2,
        max_pinned=0,
        prefer_direct_io=True,
    ) as trunk:
        initial_report = trunk.report()
        if trunk.plan.ring_slots != 2 or not initial_report["async_prefetch_enabled"]:
            raise RuntimeError(f"{phase} decoder window requires two safe ring slots")

        layer_view = trunk.bind(layers[0])
        for pos, layer in enumerate(layers):
            next_layer = layers[pos + 1] if pos + 1 < len(layers) else None
            if next_layer is not None:
                prefetch_expected += 1
                if not trunk.prefetch(next_layer):
                    raise RuntimeError(f"{phase} prefetch {layer}->{next_layer} rejected")
                prefetch_accepted += 1

            state, keepalive = streamed_state(
                layer_view,
                meta_by_layer[layer],
                weight_map,
                layer,
            )
            layer_type = text_config["layer_types"][layer]

            reference = reference_layer_from_streamed(
                reference,
                cfg,
                layer,
                state,
                reference_cache,
                position_embeddings,
                position_ids,
                full_mask,
            )

            with torch.inference_mode():
                if layer_type == "linear_attention":
                    old = linear_cache.get(layer)
                    if phase == "prefill":
                        if old is not None:
                            raise RuntimeError(f"linear cache layer {layer} already exists during prefill")
                        conv_state = None
                        recurrent_state = None
                    else:
                        if old is None:
                            raise RuntimeError(f"linear cache layer {layer} missing before decode")
                        conv_state = old["conv"]
                        recurrent_state = old["recurrent"]
                    candidate, conv_state, recurrent_state = candidate_linear_cached(
                        candidate,
                        state,
                        cfg,
                        conv_state,
                        recurrent_state,
                    )
                    linear_cache[layer] = {
                        "conv": conv_state,
                        "recurrent": recurrent_state,
                    }
                elif layer_type == "full_attention":
                    old = full_cache.get(layer)
                    if phase == "prefill":
                        if old is not None:
                            raise RuntimeError(f"full cache layer {layer} already exists during prefill")
                        key = None
                        value = None
                    else:
                        if old is None:
                            raise RuntimeError(f"full cache layer {layer} missing before decode")
                        key = old["key"]
                        value = old["value"]
                    candidate, key, value = candidate_full_cached(
                        candidate,
                        state,
                        cfg,
                        position_embeddings,
                        full_mask,
                        key,
                        value,
                    )
                    full_cache[layer] = {"key": key, "value": value}
                else:
                    raise RuntimeError(f"unsupported layer type {layer_type!r}")

            layer_cache_metrics.append(
                compare_layer_cache(
                    reference_cache,
                    linear_cache,
                    full_cache,
                    layer,
                    layer_type,
                )
            )

            state.clear()
            keepalive.clear()
            del layer_view
            if next_layer is not None:
                layer_view = trunk.bind(next_layer)

        runtime = trunk.report()

    if int(runtime["bytes_read"]) != expected_bytes:
        raise RuntimeError(
            f"{phase} decoder window {layers[0]}-{layers[-1]} read "
            f"{runtime['bytes_read']} bytes, expected {expected_bytes}"
        )
    if prefetch_accepted != prefetch_expected:
        raise RuntimeError(
            f"{phase} decoder window {layers[0]}-{layers[-1]} prefetch "
            f"{prefetch_accepted}/{prefetch_expected}"
        )

    return reference, candidate, {
        "start_layer": layers[0],
        "end_layer": layers[-1],
        "hidden_metrics": metrics(reference, candidate),
        "cache_exact": cache_metrics_exact(layer_cache_metrics),
        "runtime": runtime,
        "prefetch_expected": prefetch_expected,
        "prefetch_accepted": prefetch_accepted,
        "seconds": time.monotonic() - started,
    }, layer_cache_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--prefill-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.window_size < 2 or TOTAL_LAYERS % args.window_size:
        raise SystemExit(f"--window-size must divide {TOTAL_LAYERS} and be >=2")
    if args.prefill_len < 2:
        raise SystemExit("--prefill-len must be >=2")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    min_disk_free = shutil.disk_usage(root).free

    config, index, validated = download_metadata(model_dir)
    text_config = config["text_config"]
    cfg = Qwen3_5TextConfig(**text_config)
    cfg._attn_implementation = "eager"
    weight_map = index["weight_map"]
    global_layout = validate_global_layout(weight_map, text_config)
    layer_types = list(text_config["layer_types"])
    unsupported = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unsupported:
        raise RuntimeError(f"unsupported layer types: {unsupported}")

    # Real official embedding for prefill+decode token IDs in one pass.
    vocab_size = int(text_config["vocab_size"])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    token_ids = torch.randint(
        0,
        vocab_size,
        (1, args.prefill_len + 1),
        generator=gen,
        dtype=torch.long,
    )
    embed_shards = download_names(model_dir, weight_map, [EMBED_NAME])
    embed_bin = trunk_dir / "embedding.trunk.bin"
    embed_idx = trunk_dir / "embedding.trunk.json"
    embed_manifest = pack_named_group(
        model_dir, weight_map, [EMBED_NAME], -1, embed_bin, embed_idx
    )
    embed_meta = embed_manifest["layers"][0]
    reference_embeds = reference_embedding(model_dir, weight_map, token_ids)
    candidate_embeds, embed_runtime = streamed_embedding(
        embed_bin, embed_idx, embed_meta, token_ids
    )
    embedding_metrics = metrics(reference_embeds, candidate_embeds)
    for shard in embed_shards:
        remove_file(model_dir / shard)
    remove_file(embed_bin)
    remove_file(embed_idx)
    gc.collect()
    if not embedding_metrics["exact_equal"]:
        raise RuntimeError(f"embedding mismatch: {embedding_metrics}")

    reference_prefill = reference_embeds[:, : args.prefill_len, :].clone()
    candidate_prefill = candidate_embeds[:, : args.prefill_len, :].clone()
    reference_decode_seed = reference_embeds[:, args.prefill_len :, :].clone()
    candidate_decode_seed = candidate_embeds[:, args.prefill_len :, :].clone()
    del reference_embeds, candidate_embeds
    gc.collect()

    prefill_ids = torch.arange(args.prefill_len, dtype=torch.long).unsqueeze(0)
    decode_ids = torch.tensor([[args.prefill_len]], dtype=torch.long)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    with torch.inference_mode():
        prefill_pos = rotary(reference_prefill, prefill_ids)
        decode_pos = rotary(reference_decode_seed, decode_ids)
    prefill_mask = causal_mask(args.prefill_len)
    decode_mask = torch.zeros(
        (1, 1, 1, args.prefill_len + 1), dtype=torch.float32
    )

    reference_cache = DynamicCache(config=cfg)
    linear_cache: dict[int, dict] = {}
    full_cache: dict[int, dict] = {}
    window_specs: list[dict] = []
    prefill_windows: list[dict] = []
    prefill_cache_metrics: list[dict] = []
    total_stream_bytes = int(embed_runtime["bytes_read"])
    total_prefetch_expected = 0
    total_prefetch_accepted = 0
    retained_trunk_bytes = 0

    # Download each official source window once, pack it, delete source shards,
    # then retain only the compact K3 trunk for the later decode pass.
    for start in range(0, TOTAL_LAYERS, args.window_size):
        stop = start + args.window_size
        layers = tuple(range(start, stop))
        shards = needed_shards(weight_map, layers)
        for shard in shards:
            hf_hub_download(
                MODEL_ID,
                filename=shard,
                revision=PINNED_REVISION,
                local_dir=str(model_dir),
            )
        min_disk_free = min(min_disk_free, shutil.disk_usage(root).free)

        out_bin = trunk_dir / f"layers{start}-{stop - 1}.trunk.bin"
        out_idx = trunk_dir / f"layers{start}-{stop - 1}.trunk.json"
        manifest = pack_layers(
            model_dir,
            weight_map,
            layers,
            out_bin,
            out_idx,
            model_id=MODEL_ID,
            revision=PINNED_REVISION,
        )
        min_disk_free = min(min_disk_free, shutil.disk_usage(root).free)
        meta_by_layer = {int(item["layer"]): item for item in manifest["layers"]}
        max_layer_bytes = max(int(item["read_bytes"]) for item in manifest["layers"])
        expected_bytes = sum(int(meta_by_layer[layer]["read_bytes"]) for layer in layers)
        retained_trunk_bytes += int(manifest["packed_file_bytes"])

        for shard in shards:
            remove_file(model_dir / shard)
        gc.collect()

        spec = {
            "layers": list(layers),
            "out_bin": str(out_bin),
            "out_idx": str(out_idx),
            "meta_by_layer": meta_by_layer,
            "max_layer_bytes": max_layer_bytes,
            "expected_bytes": expected_bytes,
            "packed_file_bytes": int(manifest["packed_file_bytes"]),
        }
        window_specs.append(spec)

        reference_prefill, candidate_prefill, win, cache_items = process_window(
            phase="prefill",
            spec=spec,
            reference=reference_prefill,
            candidate=candidate_prefill,
            cfg=cfg,
            text_config=text_config,
            weight_map=weight_map,
            reference_cache=reference_cache,
            linear_cache=linear_cache,
            full_cache=full_cache,
            position_embeddings=prefill_pos,
            position_ids=prefill_ids,
            full_mask=prefill_mask,
        )
        prefill_windows.append(win)
        prefill_cache_metrics.extend(cache_items)
        total_stream_bytes += int(win["runtime"]["bytes_read"])
        total_prefetch_expected += int(win["prefetch_expected"])
        total_prefetch_accepted += int(win["prefetch_accepted"])

    prefill_decoder_metrics = metrics(reference_prefill, candidate_prefill)
    prefill_cache_bytes = cache_bytes(linear_cache, full_cache)
    prefill_reference_seq_len = int(reference_cache.get_seq_length())
    prefill_candidate_full_lengths = candidate_full_lengths(full_cache)

    # Retain official tail source shard(s) and one K3 tail until decode completes,
    # so final logits are independently checked both before and after cache reuse.
    tail_names = [FINAL_NORM_NAME, LM_HEAD_NAME]
    tail_shards = download_names(model_dir, weight_map, tail_names)
    tail_bin = trunk_dir / "tail.trunk.bin"
    tail_idx = trunk_dir / "tail.trunk.json"
    tail_manifest = pack_named_group(
        model_dir, weight_map, tail_names, 64, tail_bin, tail_idx
    )
    tail_meta = tail_manifest["layers"][0]
    min_disk_free = min(min_disk_free, shutil.disk_usage(root).free)

    reference_prefill_norm, reference_prefill_logits = reference_tail(
        reference_prefill, cfg, model_dir, weight_map
    )
    candidate_prefill_norm, candidate_prefill_logits, prefill_tail_runtime = streamed_tail(
        candidate_prefill, cfg, tail_bin, tail_idx, tail_meta
    )
    prefill_norm_metrics = metrics(reference_prefill_norm, candidate_prefill_norm)
    prefill_logits_metrics = metrics(reference_prefill_logits, candidate_prefill_logits)
    total_stream_bytes += int(prefill_tail_runtime["bytes_read"])
    del reference_prefill_norm, reference_prefill_logits
    del candidate_prefill_norm, candidate_prefill_logits
    gc.collect()

    # Cached one-token decode through all 64 retained K3 windows.
    reference_decode = reference_decode_seed
    candidate_decode = candidate_decode_seed
    decode_windows: list[dict] = []
    decode_cache_metrics: list[dict] = []
    for spec in window_specs:
        reference_decode, candidate_decode, win, cache_items = process_window(
            phase="decode",
            spec=spec,
            reference=reference_decode,
            candidate=candidate_decode,
            cfg=cfg,
            text_config=text_config,
            weight_map=weight_map,
            reference_cache=reference_cache,
            linear_cache=linear_cache,
            full_cache=full_cache,
            position_embeddings=decode_pos,
            position_ids=decode_ids,
            full_mask=decode_mask,
        )
        decode_windows.append(win)
        decode_cache_metrics.extend(cache_items)
        total_stream_bytes += int(win["runtime"]["bytes_read"])
        total_prefetch_expected += int(win["prefetch_expected"])
        total_prefetch_accepted += int(win["prefetch_accepted"])

    decode_decoder_metrics = metrics(reference_decode, candidate_decode)
    decode_cache_bytes = cache_bytes(linear_cache, full_cache)
    decode_reference_seq_len = int(reference_cache.get_seq_length())
    decode_candidate_full_lengths = candidate_full_lengths(full_cache)

    reference_decode_norm, reference_decode_logits = reference_tail(
        reference_decode, cfg, model_dir, weight_map
    )
    candidate_decode_norm, candidate_decode_logits, decode_tail_runtime = streamed_tail(
        candidate_decode, cfg, tail_bin, tail_idx, tail_meta
    )
    decode_norm_metrics = metrics(reference_decode_norm, candidate_decode_norm)
    decode_logits_metrics = metrics(reference_decode_logits, candidate_decode_logits)
    total_stream_bytes += int(decode_tail_runtime["bytes_read"])

    # Clean retained source/tail and all decoder trunks only after both passes.
    for shard in tail_shards:
        remove_file(model_dir / shard)
    remove_file(tail_bin)
    remove_file(tail_idx)
    for spec in window_specs:
        remove_file(Path(spec["out_bin"]))
        remove_file(Path(spec["out_idx"]))
    gc.collect()

    prefill_cache_exact = cache_metrics_exact(prefill_cache_metrics)
    decode_cache_exact = cache_metrics_exact(decode_cache_metrics)
    prefill_windows_exact = all(
        item["hidden_metrics"]["exact_equal"] and item["cache_exact"]
        for item in prefill_windows
    )
    decode_windows_exact = all(
        item["hidden_metrics"]["exact_equal"] and item["cache_exact"]
        for item in decode_windows
    )
    expected_linear_layers = sum(t == "linear_attention" for t in layer_types)
    expected_full_layers = sum(t == "full_attention" for t in layer_types)

    passed = all(
        [
            embedding_metrics["exact_equal"],
            prefill_windows_exact,
            decode_windows_exact,
            prefill_decoder_metrics["exact_equal"],
            decode_decoder_metrics["exact_equal"],
            prefill_cache_exact,
            decode_cache_exact,
            prefill_norm_metrics["exact_equal"],
            prefill_logits_metrics["exact_equal"],
            decode_norm_metrics["exact_equal"],
            decode_logits_metrics["exact_equal"],
            prefill_reference_seq_len == args.prefill_len,
            decode_reference_seq_len == args.prefill_len + 1,
            prefill_candidate_full_lengths == [args.prefill_len],
            decode_candidate_full_lengths == [args.prefill_len + 1],
            len(linear_cache) == expected_linear_layers,
            len(full_cache) == expected_full_layers,
            total_prefetch_accepted == total_prefetch_expected,
        ]
    )

    result = {
        "schema": "qwen38-k3-full-cached-logits-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "global_layout": global_layout,
        "token_ids": token_ids.tolist(),
        "prefill_len": args.prefill_len,
        "decode_len": 1,
        "embedding_metrics": embedding_metrics,
        "embedding_runtime": embed_runtime,
        "layer_type_counts": {
            "linear_attention": expected_linear_layers,
            "full_attention": expected_full_layers,
        },
        "prefill": {
            "decoder_windows": prefill_windows,
            "decoder_final_metrics": prefill_decoder_metrics,
            "cache_metrics": prefill_cache_metrics,
            "cache_exact": prefill_cache_exact,
            "cache_bytes": prefill_cache_bytes,
            "reference_seq_len": prefill_reference_seq_len,
            "candidate_full_cache_lengths": prefill_candidate_full_lengths,
            "final_norm_metrics": prefill_norm_metrics,
            "logits_metrics": prefill_logits_metrics,
            "tail_runtime": prefill_tail_runtime,
        },
        "decode": {
            "decoder_windows": decode_windows,
            "decoder_final_metrics": decode_decoder_metrics,
            "cache_metrics": decode_cache_metrics,
            "cache_exact": decode_cache_exact,
            "cache_bytes": decode_cache_bytes,
            "reference_seq_len": decode_reference_seq_len,
            "candidate_full_cache_lengths": decode_candidate_full_lengths,
            "final_norm_metrics": decode_norm_metrics,
            "logits_metrics": decode_logits_metrics,
            "tail_runtime": decode_tail_runtime,
        },
        "window_size": args.window_size,
        "retained_decoder_trunk_bytes": retained_trunk_bytes,
        "total_prefetch_expected": total_prefetch_expected,
        "total_prefetch_accepted": total_prefetch_accepted,
        "total_stream_bytes": total_stream_bytes,
        "min_disk_free_bytes": min_disk_free,
        "disk_free_bytes_after": shutil.disk_usage(root).free,
        "max_rss_gib": max_rss_gib(),
        "total_seconds": time.monotonic() - started,
        "reference": (
            "Transformers Qwen3_5DecoderLayer + DynamicCache loaded from K3 tensor views "
            "whose official BF16 equivalence was independently proven by prior gates; "
            "tail logits use official SafeTensors directly"
        ),
        "candidate": (
            "manual cached decoder equations over retained two-slot O_DIRECT K3Trunks "
            "+ explicit linear conv/recurrent and full-attention KV cache tensors"
        ),
        "generation_attempted": False,
        "cache_claimed": bool(passed),
        "logits_claimed": bool(passed),
        "full_cached_logits_claimed": bool(passed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
