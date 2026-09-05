#!/usr/bin/env python3
"""Exact four-layer Qwen3.8 K3 streaming chain gate (layers 0 -> 1 -> 2 -> 3).

The reference loads official Transformers decoder layers one at a time to avoid
holding four BF16 layers resident together. The candidate packs the same four
official layers into a K3 trunk, keeps only two ring slots, asynchronously
prefetches the next layer while computing the current layer, and compares the
final hidden state exactly.

This is still below token generation: no embedding table, final norm, LM head,
tokenizer, vision encoder, or prompt is used.
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
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from core import MODEL_ID, PINNED_REVISION, PREFIX
from k3_full_attention_gate import candidate_full_attention_layer, causal_mask
from k3_linear_layer_gate import (
    candidate_linear_layer,
    deterministic_hidden,
    download_metadata,
    metrics,
    source_tensor,
    streamed_tensor,
)
from k3_stream import K3Trunk, needed_shards, pack_layers

LAYERS = (0, 1, 2, 3)


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def layer_names(weight_map: dict[str, str], layer: int) -> list[str]:
    prefix = f"{PREFIX}.{layer}."
    return sorted(name for name in weight_map if name.startswith(prefix))


def local_state_from_source(model_dir: Path, weight_map: dict[str, str], layer: int) -> dict[str, torch.Tensor]:
    prefix = f"{PREFIX}.{layer}."
    return {
        name[len(prefix):]: source_tensor(model_dir, weight_map, name)
        for name in layer_names(weight_map, layer)
    }


def reference_chain(
    x: torch.Tensor,
    cfg: Qwen3_5TextConfig,
    model_dir: Path,
    weight_map: dict[str, str],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    full_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[dict]]:
    hidden = x.clone()
    evidence: list[dict] = []
    dummy = torch.empty(0, dtype=torch.float32)
    for layer in LAYERS:
        module = Qwen3_5DecoderLayer(cfg, layer).to(dtype=torch.bfloat16).eval()
        state = local_state_from_source(model_dir, weight_map, layer)
        expected = set(module.state_dict().keys())
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise RuntimeError(f"layer {layer} reference state mismatch missing={missing} unexpected={unexpected}")
        module.load_state_dict(state, strict=True)
        del state
        layer_type = cfg.layer_types[layer]
        with torch.inference_mode():
            if layer_type == "linear_attention":
                hidden = module(
                    hidden,
                    position_embeddings=(dummy, dummy),
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                )
            elif layer_type == "full_attention":
                hidden = module(
                    hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=full_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                )
            else:
                raise RuntimeError(f"unsupported layer type {layer_type!r}")
        evidence.append({"layer": layer, "type": layer_type, "rss_gib": max_rss_gib()})
        del module
        gc.collect()
    return hidden, evidence


def streamed_state(layer_view: memoryview, layer_meta: dict, weight_map: dict[str, str], layer: int) -> tuple[dict[str, torch.Tensor], list[memoryview]]:
    prefix = f"{PREFIX}.{layer}."
    state: dict[str, torch.Tensor] = {}
    keepalive: list[memoryview] = []
    for full_name in layer_names(weight_map, layer):
        tensor, view = streamed_tensor(layer_view, layer_meta, full_name)
        state[full_name[len(prefix):]] = tensor
        keepalive.append(view)
    return state, keepalive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seq_len < 2:
        raise SystemExit("--seq-len must be >= 2")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    config, index, validated = download_metadata(model_dir)
    text_config = config["text_config"]
    expected_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    actual_types = [text_config["layer_types"][layer] for layer in LAYERS]
    if actual_types != expected_types:
        raise RuntimeError(f"unexpected first four layer types: {actual_types}")
    cfg = Qwen3_5TextConfig(**text_config)
    cfg._attn_implementation = "eager"
    weight_map = index["weight_map"]

    shards = needed_shards(weight_map, LAYERS)
    for shard in shards:
        hf_hub_download(MODEL_ID, filename=shard, revision=PINNED_REVISION, local_dir=str(model_dir))

    out_bin = trunk_dir / "layers0-3.trunk.bin"
    out_idx = trunk_dir / "layers0-3.trunk.json"
    manifest = pack_layers(
        model_dir,
        weight_map,
        LAYERS,
        out_bin,
        out_idx,
        model_id=MODEL_ID,
        revision=PINNED_REVISION,
    )
    meta_by_layer = {int(item["layer"]): item for item in manifest["layers"]}
    max_layer_bytes = max(int(item["read_bytes"]) for item in manifest["layers"])

    x = deterministic_hidden(args.seq_len, args.seed)
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    with torch.inference_mode():
        position_embeddings = rotary(x, position_ids)
    full_mask = causal_mask(args.seq_len)

    reference, reference_layers = reference_chain(
        x,
        cfg,
        model_dir,
        weight_map,
        position_embeddings,
        position_ids,
        full_mask,
    )
    rss_after_reference = max_rss_gib()

    prefetch_events: list[dict] = []
    candidate = x.clone()
    budget = 2 * max_layer_bytes
    with K3Trunk(
        out_bin,
        out_idx,
        budget_bytes=budget,
        want_ring=2,
        max_pinned=0,
        prefer_direct_io=True,
    ) as trunk:
        if trunk.plan.ring_slots != 2 or not trunk.report()["async_prefetch_enabled"]:
            raise RuntimeError("chain gate requires two safe ring slots with async prefetch enabled")

        layer_view = trunk.bind(LAYERS[0])
        for pos, layer in enumerate(LAYERS):
            next_layer = LAYERS[pos + 1] if pos + 1 < len(LAYERS) else None
            accepted = trunk.prefetch(next_layer) if next_layer is not None else False
            if next_layer is not None and not accepted:
                raise RuntimeError(f"prefetch of layer {next_layer} was not accepted while layer {layer} was active")

            state, keepalive = streamed_state(layer_view, meta_by_layer[layer], weight_map, layer)
            layer_type = actual_types[pos]
            with torch.inference_mode():
                if layer_type == "linear_attention":
                    candidate = candidate_linear_layer(candidate, state, cfg)
                else:
                    candidate = candidate_full_attention_layer(
                        candidate,
                        state,
                        cfg,
                        position_embeddings,
                        full_mask,
                    )
            prefetch_events.append({
                "layer": layer,
                "next_layer": next_layer,
                "prefetch_accepted": bool(accepted),
                "rss_gib": max_rss_gib(),
            })

            state.clear()
            keepalive.clear()
            del layer_view
            if next_layer is not None:
                layer_view = trunk.bind(next_layer)
        runtime_report = trunk.report()

    result_metrics = metrics(reference, candidate)
    passed = (
        result_metrics["exact_equal"]
        and runtime_report["ring_slots"] == 2
        and runtime_report["async_prefetch_enabled"]
        and all(event["prefetch_accepted"] for event in prefetch_events[:-1])
    )
    result = {
        "schema": "qwen38-k3-four-layer-chain-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "layers": list(LAYERS),
        "layer_types": actual_types,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "source_shards": shards,
        "packed_file_bytes": manifest["packed_file_bytes"],
        "layer_read_bytes": {str(layer): int(meta_by_layer[layer]["read_bytes"]) for layer in LAYERS},
        "output_metrics": result_metrics,
        "runtime": runtime_report,
        "prefetch_events": prefetch_events,
        "reference_layers": reference_layers,
        "rss_after_reference_gib": rss_after_reference,
        "max_rss_gib": max_rss_gib(),
        "total_seconds": time.monotonic() - started,
        "reference": "official Transformers decoder layers 0-3 loaded sequentially",
        "candidate": "two-slot K3Trunk streamed layers 0-3 with next-layer async prefetch",
        "generation_attempted": False,
        "logits_claimed": False,
        "disk_free_bytes_after": shutil.disk_usage(root).free,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
