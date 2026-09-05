#!/usr/bin/env python3
"""Exact text-path logits gate over the K3-style Qwen3.8 streaming runtime.

This gate extends the already-proven decoder chain with the missing text globals:
real token embedding, all 64 decoder layers in bounded temporary windows, final
Qwen3.5 RMSNorm, and the untied BF16 LM head.  The candidate reads embedding,
decoder and head weights through page-aligned K3 trunks; the reference uses the
official Transformers decoder/RMSNorm implementation and official SafeTensors.

It intentionally computes only last-token logits (equivalent to
``logits_to_keep=1``) and does not attempt autoregressive generation or cache
reuse.  Generation is gated separately after logits equivalence is proven.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import time
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding

from core import HIDDEN, LAYERS as TOTAL_LAYERS, MODEL_ID, PINNED_REVISION
from k3_full_attention_gate import candidate_full_attention_layer, causal_mask
from k3_linear_layer_gate import (
    candidate_linear_layer,
    download_metadata,
    metrics,
    rms_norm,
    source_tensor,
    streamed_tensor,
)
from k3_stream import ALIGN, SCHEMA, TENSOR_ALIGN, K3Trunk, align_up, needed_shards, pack_layers, read_safetensors_header
from k3_windowed_chain_gate import advance_reference_layer, remove_file

EMBED_NAME = "model.language_model.embed_tokens.weight"
FINAL_NORM_NAME = "model.language_model.norm.weight"
LM_HEAD_NAME = "lm_head.weight"


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _write_zeros(handle, count: int) -> None:
    block = b"\0" * min(1024 * 1024, max(1, count))
    left = count
    while left:
        n = min(left, len(block))
        handle.write(block[:n])
        left -= n


def _copy_range(src: Path, source_offset: int, nbytes: int, dst, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    fd = os.open(src, os.O_RDONLY)
    try:
        done = 0
        while done < nbytes:
            want = min(chunk_bytes, nbytes - done)
            chunk = os.pread(fd, want, source_offset + done)
            if len(chunk) != want:
                raise IOError(f"short read from {src}: wanted {want}, got {len(chunk)}")
            dst.write(chunk)
            digest.update(chunk)
            done += want
    finally:
        os.close(fd)
    return digest.hexdigest()


def pack_named_group(
    model_dir: Path,
    weight_map: dict[str, str],
    names: list[str],
    pseudo_layer: int,
    out_bin: Path,
    out_idx: Path,
) -> dict:
    """Pack arbitrary official tensors into one K3Trunk-compatible pseudo-layer."""
    if not names:
        raise ValueError("named tensor group cannot be empty")
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise KeyError(f"missing official tensors: {missing}")

    shards = sorted({weight_map[name] for name in names})
    spans = {}
    for shard in shards:
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        spans.update(read_safetensors_header(path))

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tensors = []
    total_tensor_bytes = 0
    with out_bin.open("wb") as dst:
        start = align_up(dst.tell(), ALIGN)
        if start > dst.tell():
            _write_zeros(dst, start - dst.tell())
        for name in sorted(names):
            span = spans.get(name)
            if span is None:
                raise KeyError(f"{name} missing from downloaded SafeTensors headers")
            expected_shard = str(weight_map[name])
            if span.shard != expected_shard:
                raise ValueError(f"{name}: index={expected_shard}, header={span.shard}")
            tensor_at = align_up(dst.tell() - start, TENSOR_ALIGN)
            absolute = start + tensor_at
            if absolute > dst.tell():
                _write_zeros(dst, absolute - dst.tell())
            digest = _copy_range(model_dir / span.shard, span.source_offset, span.nbytes, dst)
            tensors.append({
                "name": name,
                "dtype": span.dtype,
                "shape": list(span.shape),
                "offset": tensor_at,
                "nbytes": span.nbytes,
                "source_shard": span.shard,
                "source_offset": span.source_offset,
                "sha256": digest,
            })
            total_tensor_bytes += span.nbytes
        data_bytes = dst.tell() - start
        end = align_up(dst.tell(), ALIGN)
        if end > dst.tell():
            _write_zeros(dst, end - dst.tell())

    layer_meta = {
        "layer": pseudo_layer,
        "file_offset": start,
        "data_bytes": data_bytes,
        "read_bytes": end - start,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    manifest = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "alignment": ALIGN,
        "tensor_alignment": TENSOR_ALIGN,
        "layers": [layer_meta],
        "total_tensor_bytes": total_tensor_bytes,
        "packed_file_bytes": out_bin.stat().st_size,
    }
    out_idx.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def download_names(model_dir: Path, weight_map: dict[str, str], names: list[str]) -> list[str]:
    shards = sorted({str(weight_map[name]) for name in names})
    for shard in shards:
        hf_hub_download(
            MODEL_ID,
            filename=shard,
            revision=PINNED_REVISION,
            local_dir=str(model_dir),
        )
    return shards


def validate_global_layout(weight_map: dict[str, str], text_config: dict) -> dict:
    names = [EMBED_NAME, FINAL_NORM_NAME, LM_HEAD_NAME]
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise KeyError(f"official global tensor(s) missing: {missing}")
    return {
        "embedding": EMBED_NAME,
        "final_norm": FINAL_NORM_NAME,
        "lm_head": LM_HEAD_NAME,
        "vocab_size": int(text_config["vocab_size"]),
        "hidden_size": int(text_config["hidden_size"]),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
        "embedding_shard": str(weight_map[EMBED_NAME]),
        "final_norm_shard": str(weight_map[FINAL_NORM_NAME]),
        "lm_head_shard": str(weight_map[LM_HEAD_NAME]),
    }


def reference_embedding(model_dir: Path, weight_map: dict[str, str], token_ids: torch.Tensor) -> torch.Tensor:
    weight = source_tensor(model_dir, weight_map, EMBED_NAME)
    if weight.shape[1] != HIDDEN:
        raise ValueError(f"embedding hidden dim mismatch: {tuple(weight.shape)}")
    with torch.inference_mode():
        out = F.embedding(token_ids, weight)
    del weight
    gc.collect()
    return out


def streamed_embedding(
    out_bin: Path,
    out_idx: Path,
    layer_meta: dict,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    read_bytes = int(layer_meta["read_bytes"])
    with K3Trunk(
        out_bin,
        out_idx,
        budget_bytes=read_bytes,
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as trunk:
        view = trunk.bind(-1)
        weight, keepalive = streamed_tensor(view, layer_meta, EMBED_NAME)
        with torch.inference_mode():
            out = F.embedding(token_ids, weight)
        report = trunk.report()
        del weight, keepalive, view
    if report["async_prefetch_enabled"]:
        raise RuntimeError("single-slot embedding reader must not enable async prefetch")
    return out, report


def reference_tail(
    hidden: torch.Tensor,
    cfg: Qwen3_5TextConfig,
    model_dir: Path,
    weight_map: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_module = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dtype=torch.bfloat16).eval()
    norm_weight = source_tensor(model_dir, weight_map, FINAL_NORM_NAME)
    norm_module.load_state_dict({"weight": norm_weight}, strict=True)
    del norm_weight
    with torch.inference_mode():
        normalized = norm_module(hidden)
    del norm_module
    gc.collect()

    head = source_tensor(model_dir, weight_map, LM_HEAD_NAME)
    if tuple(head.shape) != (int(cfg.vocab_size), int(cfg.hidden_size)):
        raise ValueError(f"lm_head shape mismatch: {tuple(head.shape)}")
    with torch.inference_mode():
        logits = F.linear(normalized[:, -1:, :], head)
    del head
    gc.collect()
    return normalized, logits


def streamed_tail(
    hidden: torch.Tensor,
    cfg: Qwen3_5TextConfig,
    out_bin: Path,
    out_idx: Path,
    layer_meta: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    read_bytes = int(layer_meta["read_bytes"])
    with K3Trunk(
        out_bin,
        out_idx,
        budget_bytes=read_bytes,
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as trunk:
        view = trunk.bind(64)
        norm_weight, norm_keepalive = streamed_tensor(view, layer_meta, FINAL_NORM_NAME)
        head, head_keepalive = streamed_tensor(view, layer_meta, LM_HEAD_NAME)
        with torch.inference_mode():
            normalized = rms_norm(hidden, norm_weight, cfg.rms_norm_eps)
            logits = F.linear(normalized[:, -1:, :], head)
        report = trunk.report()
        del norm_weight, head, norm_keepalive, head_keepalive, view
    if report["async_prefetch_enabled"]:
        raise RuntimeError("single-slot LM tail reader must not enable async prefetch")
    return normalized, logits, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.window_size < 2 or TOTAL_LAYERS % args.window_size:
        raise SystemExit(f"--window-size must divide {TOTAL_LAYERS} and be >=2")
    if args.seq_len < 2:
        raise SystemExit("--seq-len must be >=2")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    config, index, validated = download_metadata(model_dir)
    text_config = config["text_config"]
    cfg = Qwen3_5TextConfig(**text_config)
    cfg._attn_implementation = "eager"
    weight_map = index["weight_map"]
    global_layout = validate_global_layout(weight_map, text_config)
    vocab_size = int(text_config["vocab_size"])

    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    token_ids = torch.randint(0, vocab_size, (1, args.seq_len), generator=gen, dtype=torch.long)

    # 1) Real official token embedding: SafeTensors reference vs one-slot K3 pseudo-layer.
    embed_shards = download_names(model_dir, weight_map, [EMBED_NAME])
    embed_bin = trunk_dir / "embedding.trunk.bin"
    embed_idx = trunk_dir / "embedding.trunk.json"
    embed_manifest = pack_named_group(model_dir, weight_map, [EMBED_NAME], -1, embed_bin, embed_idx)
    embed_meta = embed_manifest["layers"][0]
    reference = reference_embedding(model_dir, weight_map, token_ids)
    candidate, embed_runtime = streamed_embedding(embed_bin, embed_idx, embed_meta, token_ids)
    embedding_metrics = metrics(reference, candidate)
    if not embedding_metrics["exact_equal"]:
        raise RuntimeError(f"embedding mismatch: {embedding_metrics}")
    for shard in embed_shards:
        remove_file(model_dir / shard)
    remove_file(embed_bin)
    remove_file(embed_idx)
    gc.collect()

    # Shared position inputs exactly matching text-only positions for all decoder windows.
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    with torch.inference_mode():
        position_embeddings = rotary(reference, position_ids)
    full_mask = causal_mask(args.seq_len)

    windows = []
    total_stream_bytes = int(embed_runtime["bytes_read"])
    total_prefetch_expected = 0
    total_prefetch_accepted = 0
    decoder_exact = True

    for start in range(0, TOTAL_LAYERS, args.window_size):
        stop = start + args.window_size
        layers = tuple(range(start, stop))
        window_started = time.monotonic()
        shards = needed_shards(weight_map, layers)
        for shard in shards:
            hf_hub_download(
                MODEL_ID,
                filename=shard,
                revision=PINNED_REVISION,
                local_dir=str(model_dir),
            )

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
        meta_by_layer = {int(item["layer"]): item for item in manifest["layers"]}
        max_layer_bytes = max(int(item["read_bytes"]) for item in manifest["layers"])
        expected_bytes = sum(int(meta_by_layer[layer]["read_bytes"]) for layer in layers)

        for layer in layers:
            reference = advance_reference_layer(
                reference,
                cfg,
                model_dir,
                weight_map,
                layer,
                position_embeddings,
                position_ids,
                full_mask,
            )

        for shard in shards:
            remove_file(model_dir / shard)
        gc.collect()

        with K3Trunk(
            out_bin,
            out_idx,
            budget_bytes=2 * max_layer_bytes,
            want_ring=2,
            max_pinned=0,
            prefer_direct_io=True,
        ) as trunk:
            if trunk.plan.ring_slots != 2 or not trunk.report()["async_prefetch_enabled"]:
                raise RuntimeError("decoder window requires two safe ring slots")
            layer_view = trunk.bind(layers[0])
            for pos, layer in enumerate(layers):
                next_layer = layers[pos + 1] if pos + 1 < len(layers) else None
                accepted = trunk.prefetch(next_layer) if next_layer is not None else False
                if next_layer is not None:
                    total_prefetch_expected += 1
                    if not accepted:
                        raise RuntimeError(f"prefetch {layer}->{next_layer} rejected")
                    total_prefetch_accepted += 1

                prefix = f"model.language_model.layers.{layer}."
                state = {}
                keepalive = []
                for full_name in sorted(name for name in weight_map if name.startswith(prefix)):
                    tensor, keep = streamed_tensor(layer_view, meta_by_layer[layer], full_name)
                    state[full_name[len(prefix):]] = tensor
                    keepalive.append(keep)

                layer_type = text_config["layer_types"][layer]
                with torch.inference_mode():
                    if layer_type == "linear_attention":
                        candidate = candidate_linear_layer(candidate, state, cfg)
                    elif layer_type == "full_attention":
                        candidate = candidate_full_attention_layer(
                            candidate,
                            state,
                            cfg,
                            position_embeddings,
                            full_mask,
                        )
                    else:
                        raise RuntimeError(f"unsupported layer type {layer_type!r}")
                state.clear()
                keepalive.clear()
                del layer_view
                if next_layer is not None:
                    layer_view = trunk.bind(next_layer)
            runtime = trunk.report()

        if int(runtime["bytes_read"]) != expected_bytes:
            raise RuntimeError(
                f"decoder window {start}-{stop - 1} read {runtime['bytes_read']}, expected {expected_bytes}"
            )
        total_stream_bytes += int(runtime["bytes_read"])
        window_metrics = metrics(reference, candidate)
        decoder_exact = decoder_exact and window_metrics["exact_equal"]
        windows.append({
            "start_layer": start,
            "end_layer": stop - 1,
            "metrics": window_metrics,
            "runtime": runtime,
            "seconds": time.monotonic() - window_started,
        })
        remove_file(out_bin)
        remove_file(out_idx)
        gc.collect()
        if not window_metrics["exact_equal"]:
            break

    if not decoder_exact:
        raise RuntimeError("decoder chain ceased to be exact")

    # 3) Official final RMSNorm + untied LM head, read as one one-slot K3 tail group.
    tail_names = [FINAL_NORM_NAME, LM_HEAD_NAME]
    tail_shards = download_names(model_dir, weight_map, tail_names)
    tail_bin = trunk_dir / "tail.trunk.bin"
    tail_idx = trunk_dir / "tail.trunk.json"
    tail_manifest = pack_named_group(model_dir, weight_map, tail_names, 64, tail_bin, tail_idx)
    tail_meta = tail_manifest["layers"][0]

    reference_norm, reference_logits = reference_tail(reference, cfg, model_dir, weight_map)
    candidate_norm, candidate_logits, tail_runtime = streamed_tail(candidate, cfg, tail_bin, tail_idx, tail_meta)
    norm_metrics = metrics(reference_norm, candidate_norm)
    logits_metrics = metrics(reference_logits, candidate_logits)
    total_stream_bytes += int(tail_runtime["bytes_read"])

    for shard in tail_shards:
        remove_file(model_dir / shard)
    remove_file(tail_bin)
    remove_file(tail_idx)
    gc.collect()

    passed = (
        embedding_metrics["exact_equal"]
        and decoder_exact
        and norm_metrics["exact_equal"]
        and logits_metrics["exact_equal"]
        and total_prefetch_accepted == total_prefetch_expected
        and len(windows) == TOTAL_LAYERS // args.window_size
    )
    result = {
        "schema": "qwen38-k3-logits-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "global_layout": global_layout,
        "token_ids": token_ids.tolist(),
        "seq_len": args.seq_len,
        "logits_to_keep": 1,
        "embedding_metrics": embedding_metrics,
        "embedding_runtime": embed_runtime,
        "decoder_windows": windows,
        "decoder_final_metrics": metrics(reference, candidate),
        "final_norm_metrics": norm_metrics,
        "logits_metrics": logits_metrics,
        "tail_runtime": tail_runtime,
        "total_prefetch_expected": total_prefetch_expected,
        "total_prefetch_accepted": total_prefetch_accepted,
        "total_stream_bytes": total_stream_bytes,
        "max_rss_gib": max_rss_gib(),
        "disk_free_bytes_after": shutil.disk_usage(root).free,
        "total_seconds": time.monotonic() - started,
        "reference": "official BF16 SafeTensors + Transformers Qwen3_5 decoder/RMSNorm + nn.Linear-equivalent F.linear",
        "candidate": "K3 trunks for embedding, 64 decoder layers, final norm and LM head",
        "generation_attempted": False,
        "logits_claimed": bool(passed),
        "cache_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
